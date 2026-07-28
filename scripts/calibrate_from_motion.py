#!/usr/bin/env python3
"""Calibrate an ALREADY-RECORDED video from vehicle motion + GPS speed.

WHY THIS EXISTS
scripts/calibrate.py needs two markers placed on the ground at measured
distances. That is the accurate method, but it only works if you still have the
car and the camera in the same pose. For footage already shot -- someone else's
clip, a dataset, last week's drive -- it is useless.

This recovers calibration from the video itself.

THE IDEA
For a point on the ground at forward distance Z, the pixel row is

    v = cy + f*h/Z        so        Z = f*h / (v - cy)

Drive forward at speed s for dt and a fixed road feature moves from Z1 to
Z2 = Z1 - s*dt. Track that feature between two frames and you get

    s*dt  =  f*h * ( 1/(v1-cy) - 1/(v2-cy) )

Every tracked feature is one equation. GPS supplies s. Fit over many features
and you recover TWO things: the product (f*h) and the horizon row cy.

WHY THE PRODUCT IS ENOUGH
Forward range is exactly  Z = f*h/(v-cy)  -- it depends on f and h only through
their product. Once (f*h) and cy are measured, every forward distance the
pipeline reports is correct no matter how you split that product. You still pick
an h to write a config (f = product/h, vfov = 2*atan((H/2)/f)), and that choice
does affect LATERAL offsets, which scale as 1/f. Forward range, the dominant
term, is pinned by measurement.

LIMITS -- read these
  - Needs the vehicle actually moving. Stationary sections contribute nothing.
  - Assumes a flat road across the tracked interval.
  - Tracks road-SURFACE features. A moving vehicle in the ROI would corrupt the
    fit, so keep the ROI on bare road and check the residual.
  - GPS speed carries its own error; smoothing helps but is not magic.

USAGE
    uv run python scripts/calibrate_from_motion.py \
        --video road_videos/test_1/video.mp4 \
        --gpx   road_videos/test_1/track.gpx \
        --height 1.2 --t-start 8 --t-end 24 \
        --roi-top 0.42 --roi-bottom 0.66 \
        --write configs/camera/car_dash.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pos.geo import haversine_m  # noqa: E402
from pos.ingest import load_gpx, probe_duration, probe_fps  # noqa: E402


def grab(video: Path, t: float, out: Path) -> np.ndarray | None:
    """Decode one frame at time t as a greyscale array."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1", str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out.exists():
        return None
    from PIL import Image

    with Image.open(out) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def speed_at(track, t: float) -> float:
    """Smoothed GPS speed (m/s) around time t."""
    if len(track) < 2:
        return 0.0
    best = min(range(len(track)), key=lambda i: abs(track[i].t - t))
    lo = max(0, best - 1)
    hi = min(len(track) - 1, best + 1)
    if hi <= lo:
        return 0.0
    dist = sum(
        haversine_m(track[i].lat, track[i].lon, track[i + 1].lat, track[i + 1].lon)
        for i in range(lo, hi)
    )
    dt = track[hi].t - track[lo].t
    return dist / dt if dt > 0 else 0.0


def track_rows(
    a: np.ndarray,
    b: np.ndarray,
    roi_top: float,
    roi_bottom: float,
    patch: int = 24,
    max_shift: int = 220,
    exclude: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """Match road patches from frame a into frame b. Returns [(v1, v2), ...].

    Ground features move DOWN the image as the vehicle approaches, so the search
    only looks downward. Matching is normalised cross-correlation down a single
    column strip, which suffices because forward motion is almost purely vertical
    in image space near the centre.
    """
    H, W = a.shape
    y0, y1 = int(H * roi_top), int(H * roi_bottom)
    out: list[tuple[float, float]] = []

    ex0 = int(W * exclude[0]) if exclude else -1
    ex1 = int(W * exclude[1]) if exclude else -1

    for cx in range(int(W * 0.18), int(W * 0.82), 32):
        if ex0 <= cx <= ex1:
            continue  # a vehicle ahead moves with us: near-zero parallax
        for v1 in range(y0, max(y0 + 1, y1 - patch), 18):
            tpl = a[v1 : v1 + patch, cx - patch : cx + patch]
            if tpl.shape[0] < patch or tpl.std() < 9.0:
                continue  # featureless asphalt: nothing to lock onto

            tpl_z = (tpl - tpl.mean()) / (tpl.std() + 1e-6)
            best_score, best_v = -2.0, None
            for v2 in range(v1, min(H - patch, v1 + max_shift)):
                win = b[v2 : v2 + patch, cx - patch : cx + patch]
                if win.shape != tpl.shape:
                    continue
                sd = win.std()
                if sd < 5.0:
                    continue
                score = float((tpl_z * ((win - win.mean()) / (sd + 1e-6))).mean())
                if score > best_score:
                    best_score, best_v = score, v2

            # Require a confident, non-trivial downward move.
            if best_v is not None and best_score > 0.72 and best_v > v1 + 3:
                out.append((v1 + patch / 2.0, best_v + patch / 2.0))
    return out


def _pair_obs(job: tuple) -> tuple[int, float, float, list[tuple[float, float]]]:
    """One frame pair, start to finish. Module level so it can be pickled.

    Pairs are independent -- two ffmpeg decodes and a correlation search each --
    and track_rows() is a pure-Python triple loop, so the GIL makes threads
    useless here. Processes give the real speedup. Every worker writes to its
    own temp files, or the pairs would overwrite each other's frames.
    """
    idx, video, t, dt, roi_top, roi_bottom, exclude, dist, tmpdir = job
    tmp = Path(tmpdir)
    a = grab(video, t, tmp / f"a{idx}.png")
    b = grab(video, t + dt, tmp / f"b{idx}.png")
    if a is None or b is None:
        return idx, t, dist, []
    return idx, t, dist, track_rows(a, b, roi_top, roi_bottom, exclude=exclude)


def fit(
    obs: list[tuple[float, float, float]],
    H: int,
    cy_max: float | None = None,
    cy_fixed: float | None = None,
):
    """Least-squares fit of (product = f*h, cy) to observations (v1, v2, dist).

    Grid-search cy; for each candidate the model is linear in the product, so
    solve that in closed form:
        dist = product * (1/(v1-cy) - 1/(v2-cy))
    Returns (product, cy, rms_residual_m) or (None, None, inf).
    """
    best: tuple[float | None, float | None, float] = (None, None, float("inf"))

    # The horizon MUST lie above every tracked road feature. Without this bound
    # the search happily places cy inside the ROI, fits a sliver of rows, and
    # returns a nonsense focal length (observed: cy=636 inside a 475-713 ROI,
    # giving vfov 151 deg).
    hi = min(cy_max if cy_max is not None else H * 0.60, H * 0.60)
    # A camera aimed steeply down -- bike, helmet or chest mount -- can put the
    # horizon at the very top of the frame or off it entirely, so cy may be small
    # or even negative. Starting the search at 0.10*H pinned a real Kohima bike
    # clip against the floor and returned a bogus 20 deg vfov.
    lo = -H * 0.60

    # Fitting BOTH the product and cy needs tracked features spread over a wide
    # band of rows, so the curvature of 1/(v-cy) is visible. At low speed only
    # near features move enough to track, they all land in a narrow band, and cy
    # becomes unidentifiable -- the search then runs to whichever bound it
    # started from (seen on a 2.3 m/s bike clip: cy pinned at both -648 and
    # +108, giving vfov 2.7 and 20.1 deg respectively). Supplying a measured
    # horizon collapses this to a single well-conditioned unknown.
    grid = [cy_fixed] if cy_fixed is not None else np.arange(lo, hi, 0.5)
    for cy in grid:
        k, d = [], []
        for v1, v2, dist in obs:
            if v1 - cy < 12 or v2 - cy < 12:
                continue  # too near the horizon: 1/(v-cy) explodes
            k.append(1.0 / (v1 - cy) - 1.0 / (v2 - cy))
            d.append(dist)
        if len(k) < 12:
            continue
        k_arr, d_arr = np.asarray(k), np.asarray(d)
        denom = float((k_arr * k_arr).sum())
        if denom <= 1e-12:
            continue
        product = float((k_arr * d_arr).sum() / denom)
        if product <= 0:
            continue
        resid = float(np.sqrt(np.mean((product * k_arr - d_arr) ** 2)))
        if resid < best[2]:
            best = (product, float(cy), resid)

    return best


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--gpx", type=Path, required=True)
    p.add_argument("--height", type=float, default=1.2,
                   help="Assumed camera height (m). Affects lateral scale only.")
    p.add_argument("--t-start", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=None,
                   help="Defaults to the video duration. Left unbounded it used "
                        "to spread the pairs over 1e9 seconds and find nothing.")
    p.add_argument("--dt", type=float, default=0.30,
                   help="Gap between the two frames of each pair, seconds.")
    p.add_argument("--pairs", type=int, default=14)
    p.add_argument("--min-speed", type=float, default=3.0,
                   help="Skip pairs slower than this (m/s).")
    p.add_argument("--roi-top", type=float, default=0.42,
                   help="Top of the road region, fraction of frame height.")
    p.add_argument("--roi-bottom", type=float, default=0.66,
                   help="Bottom of the road region -- keep ABOVE the bonnet.")
    p.add_argument("--time-offset", type=float, default=0.0,
                   help="Seconds to add to video time to reach GPS time.")
    p.add_argument("--exclude-cols", type=str, default="",
                   help="Column band to ignore, e.g. 0.42,0.70 -- put the "
                        "vehicle ahead in here so its zero parallax is skipped.")
    p.add_argument("--horizon", type=float, default=None,
                   help="Measured horizon row (px). Supply this when the fit "
                        "cannot identify it -- typical at low speed. Read it off "
                        "a frame: the row where the road vanishes.")
    p.add_argument("--max-range", type=float, default=60.0)
    p.add_argument("--write", type=Path)
    p.add_argument("--json", type=Path, dest="json_out",
                   help="Write the result as JSON here, including any warnings. "
                        "Written even on failure, with ok=false and a reason, so "
                        "a caller can decide what to do instead of scraping stdout.")
    p.add_argument("--workers", type=int, default=0,
                   help="Frame pairs to process at once. 0 = auto (cores-1, "
                        "capped at 8). 1 forces the serial path.")
    args = p.parse_args()

    warnings: list[str] = []

    def finish(ok: bool, error: str | None = None, **fields) -> None:
        """Write the JSON sidecar, if one was asked for."""
        if not args.json_out:
            return
        payload = {"ok": ok, "error": error, "warnings": warnings}
        payload.update(fields)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")

    def bail(msg: str) -> "SystemExit":
        finish(False, error=msg)
        return SystemExit(msg)

    exclude = None
    if args.exclude_cols:
        a_, b_ = args.exclude_cols.split(",")
        exclude = (float(a_), float(b_))

    tmp = Path(tempfile.mkdtemp())
    track, _ = load_gpx(args.gpx)
    src_fps = probe_fps(args.video)

    # The last pair needs --dt of video after it, so stop short of the end.
    duration = probe_duration(args.video)
    if args.t_end is None:
        if duration <= 0:
            raise bail(
                "could not probe the video duration; pass --t-end explicitly"
            )
        args.t_end = max(args.t_start + 1.0, duration - args.dt - 0.5)

    first = grab(args.video, args.t_start + 1.0, tmp / "probe.png")
    if first is None:
        raise bail("could not decode a frame; is --video readable?")
    H, W = first.shape
    print(f"frame {W}x{H}, source {src_fps:.2f} fps")
    print(f"road ROI rows {int(H * args.roi_top)}-{int(H * args.roi_bottom)} "
          "(must stay above the bonnet)")
    print()

    obs: list[tuple[float, float, float]] = []
    span = max(args.t_end - args.t_start, 1.0)
    step = span / max(args.pairs, 1)

    # Decide the work up front: GPS speed is cheap to look up, and a pair below
    # the speed floor contributes nothing, so it never reaches a worker.
    jobs: list[tuple] = []
    for i in range(args.pairs):
        t = args.t_start + i * step
        s = speed_at(track, t + args.time_offset)
        if s < args.min_speed:
            print(f"  t={t:5.1f}s  speed {s:4.1f} m/s  -- skipped (too slow)")
            continue
        jobs.append(
            (i, args.video, t, args.dt, args.roi_top, args.roi_bottom,
             exclude, s * args.dt, str(tmp))
        )

    n_workers = args.workers if args.workers > 0 else min(8, max(1, (os.cpu_count() or 2) - 1))
    n_workers = max(1, min(n_workers, len(jobs) or 1))
    if n_workers > 1 and jobs:
        print(f"  {n_workers} pairs at a time")

    if n_workers == 1:
        results = [_pair_obs(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_pair_obs, jobs))

    for _, t, dist, rows in sorted(results, key=lambda r: r[0]):
        obs.extend((v1, v2, dist) for v1, v2 in rows)
        print(f"  t={t:5.1f}s  moved {dist:5.2f} m  -> {len(rows)} tracks")

    print(f"\n{len(obs)} tracked features total")
    if len(obs) < 20:
        raise bail(
            "Not enough tracks to fit. Try a longer or faster window, a larger "
            "--dt, or widen the ROI (but stay above the bonnet)."
        )

    cy_max = args.roi_top * H - 15.0   # horizon strictly above the road ROI
    cy_fixed = args.horizon if args.horizon is not None else None
    product, cy, resid = fit(obs, H, cy_max=cy_max, cy_fixed=cy_fixed)
    if product is None or cy is None:
        raise bail("fit failed; check the ROI bounds and the speed window")

    # Robust pass: drop the worst 20% of residuals and refit.
    k = np.array([1.0 / (v1 - cy) - 1.0 / (v2 - cy) for v1, v2, _ in obs])
    d = np.array([dd for _, _, dd in obs])
    err = np.abs(product * k - d)
    obs2 = [o for o, keep in zip(obs, err <= np.quantile(err, 0.80)) if keep]
    product, cy, resid = fit(obs2, H, cy_max=cy_max, cy_fixed=cy_fixed)
    if product is None or cy is None:
        raise bail("refit failed after outlier rejection")

    # Warn if the optimum sits on a search boundary: that is not a real minimum.
    lo_bound, hi_bound = -H * 0.60, min(cy_max, H * 0.60)
    on_edge = cy_fixed is None and (
        (cy - lo_bound) < 1.0 or (hi_bound - cy) < 1.0
    )

    f_px = product / args.height
    vfov = 2 * math.degrees(math.atan((H / 2) / f_px))
    hfov = 2 * math.degrees(math.atan((W / 2) / f_px))
    pitch = (cy - H / 2) / H

    print("\nFIT (after dropping the worst 20% of residuals)")
    print(f"  inliers            : {len(obs2)}")
    print(f"  f*h product        : {product:.1f} px*m")
    print(f"  horizon row cy     : {cy:.1f}  of {H}")
    print(f"  rms residual       : {resid:.2f} m")
    print()
    print(f"  assuming height_m  = {args.height}")
    print(f"  -> focal length    : {f_px:.1f} px")
    print(f"  -> vfov_deg        : {vfov:.2f}")
    print(f"  -> pitch_offset    : {pitch:+.4f}")
    print()
    print("  Forward ranges depend only on (f*h) and cy, both measured above, so")
    print("  they hold whatever height you assumed. Lateral offsets scale as 1/f,")
    print("  so those do rely on the height being roughly right.")

    # Collected as data, not just printed: an automated caller (the studio job)
    # has to be able to reject a bad fit rather than silently shipping it.
    if on_edge:
        warnings.append(
            f"fitted horizon {cy:.1f} sits on a search boundary, so this is not a "
            "true optimum -- widen the ROI or check the tracked region is road"
        )
    if not (20.0 < vfov < 140.0):
        warnings.append(
            f"vfov {vfov:.1f} deg is implausible -- the fit probably locked onto "
            "moving traffic; narrow the ROI to bare road"
        )
    if resid > 1.0:
        warnings.append(
            f"residual {resid:.2f} m is high -- suspect a slope, poor GPS, or "
            "tracks on a moving vehicle"
        )
    for w in warnings:
        print(f"\n  WARNING: {w}")

    print("\n  row -> forward range under this calibration")
    for frac in (0.45, 0.50, 0.55, 0.60, 0.65):
        v = H * frac
        if v - cy > 1:
            print(f"    y={v:6.0f}  ->  {product / (v - cy):6.1f} m")

    yaml_text = (
        "# Solved by scripts/calibrate_from_motion.py from vehicle motion + GPS.\n"
        f"# f*h product = {product:.1f} px*m, horizon row = {cy:.1f} of {H},\n"
        f"# rms residual {resid:.2f} m over {len(obs2)} tracked features.\n"
        "# Forward range depends only on the product and the horizon, both measured.\n"
        f"height_m: {args.height}\n"
        f"vfov_deg: {vfov:.2f}\n"
        f"hfov_deg: {hfov:.2f}\n"
        f"pitch_offset_frac: {pitch:.4f}\n"
        f"max_range_m: {args.max_range}\n"
    )
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(yaml_text)
        print(f"\nWrote {args.write}   (use --camera {args.write.stem})")
    else:
        print("\nPaste into configs/camera/<name>.yaml:\n")
        print(yaml_text)

    finish(
        True,
        vfov_deg=round(vfov, 2),
        hfov_deg=round(hfov, 2),
        height_m=args.height,
        pitch_offset_frac=round(pitch, 4),
        product=round(product, 1),
        horizon_row=round(cy, 1),
        residual_m=round(resid, 2),
        inliers=len(obs2),
        frame=[W, H],
        wrote=str(args.write) if args.write else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
