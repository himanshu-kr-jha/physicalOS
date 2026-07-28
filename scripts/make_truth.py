#!/usr/bin/env python3
"""Author hand-placed detections for a REAL video, and check they land right.

Why this exists: `pos perceive --backend mock --truth f.json` replays a fixture
file instead of calling a detector, and everything downstream -- projection,
triangulation, clustering, scoring -- then runs for real on top of it. That is
the honest way to script a demo: the geometry is genuine, only the perception
step is hand-authored. But the fixture format is keyed by exact keyframe
timestamp and uses boxes normalised to 0..1000, which is unwriteable by hand.

So you write a spec in SECONDS and PIXELS, marking a defect at the two or three
moments you can see it clearly, and this interpolates it across every keyframe
in between:

    defects:
      - cls: pothole
        severity: 4
        confidence: 0.93
        evidence: "Edge-broken pothole ~40 cm across, left wheel path."
        track:
          - {t: 12.0, box: [820, 690, 900, 730]}   # far, small
          - {t: 13.5, box: [640, 780, 1180, 980]}  # close, large

Marking a defect across SEVERAL keyframes is not cosmetic. One sighting gives a
single ground-plane projection, which is pitch-sensitive and can only claim
"+/-2-4 m". Enough sightings, from far enough apart, let the clusterer intersect
real bearing rays and the finding comes out `triangulated` with a genuine
residual in metres. `cluster` demands >= 8 degrees of parallax before it will
trust that (pos/cluster.py, min_parallax_deg), and two rules decide whether you
clear it -- both measured on a real run:

  1. Mark from FIRST SIGHTING, not from where the defect is obvious. Starting
     when it is a small shape in the distance and following it past the camera
     is what builds the baseline.
  2. Keep the box bottom INSIDE the frame. The bottom edge is the ranging cue;
     once it is clipped by the frame edge, every sighting ranges as "~3 m ahead"
     and the estimate travels with the camera, so the rays come out nearly
     parallel.

Same pile, same video: a 3-keypoint track over 3.0s with the box running off
the bottom edge gave parallax 3.56 deg and stayed `ground_plane`. Extending it
back to first sighting (5.4s) and keeping the bottom edge in frame gave parallax
26.23 deg, `triangulated`, residual 0.91 m.

Commands
--------
    frames   list the run's keyframes, so you know which timestamps exist
    grid     write one keyframe with a labelled pixel grid, to read boxes off
    build    spec (seconds + pixels) -> truth.json (timestamps + 0..1000)
    preview  draw a truth.json back onto the keyframes, to check alignment

Typical session
---------------
    uv run pos ingest --video drive.mp4 --gpx drive.gpx --out runs/demo --fps 2
    uv run python scripts/make_truth.py frames  --run runs/demo
    uv run python scripts/make_truth.py grid    --run runs/demo --t 12.0 --out grid.jpg
    #   ... write spec.yaml ...
    uv run python scripts/make_truth.py build   --run runs/demo --spec spec.yaml --out truth.json
    uv run python scripts/make_truth.py preview --run runs/demo --truth truth.json --out preview/
    uv run pos perceive --run runs/demo --backend mock --truth truth.json
    uv run pos localize --run runs/demo
    uv run pos cluster  --run runs/demo
    uv run pos score    --run runs/demo

Note `pos ingest` is run ONCE and the later stages are run individually, rather
than `pos run`. Re-ingesting would resample the video, and the spec is written
against the timestamps that sampling produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pos.config import DomainConfig  # noqa: E402
from pos.schema import BOX_SCALE  # noqa: E402

# A point defect whose box bottom sits high in the frame is above the horizon,
# where `localize` refuses to range it. 0.52 is just below the usual horizon for
# a level dashcam; boxes above it are almost always a misread coordinate.
HORIZON_FRAC = 0.52


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_frames(run: Path) -> list[dict[str, Any]]:
    path = run / "frames.json"
    if not path.exists():
        sys.exit(f"{path} not found. Run `pos ingest --out {run}` first.")
    frames = json.loads(path.read_text())
    if not frames:
        sys.exit(f"{path} is empty.")
    return sorted(frames, key=lambda f: f["t_sec"])


def load_domain(run: Path) -> DomainConfig:
    manifest = run / "manifest.json"
    key = "road"
    if manifest.exists():
        key = json.loads(manifest.read_text()).get("domain", "road")
    return DomainConfig.load(key)


def frame_at(frames: list[dict[str, Any]], t: float) -> dict[str, Any]:
    """The keyframe nearest `t`. Used by `grid`, where exactness does not matter."""
    return min(frames, key=lambda f: abs(f["t_sec"] - t))


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------


def cmd_frames(args: argparse.Namespace) -> None:
    frames = load_frames(args.run)
    lo = args.t_from if args.t_from is not None else -1e9
    hi = args.t_to if args.t_to is not None else 1e9

    shown = [f for f in frames if lo <= f["t_sec"] <= hi][:: max(args.every, 1)]

    print(f"{len(frames)} keyframes, {frames[0]['t_sec']:.2f}s .. {frames[-1]['t_sec']:.2f}s")
    print(f"{'t_sec':>9}  {'frame':>7}  {'size':>11}  path")
    for f in shown:
        size = f"{f['width']}x{f['height']}"
        print(f"{f['t_sec']:9.2f}  {f['frame_id']:>7}  {size:>11}  {f['path']}")
    if len(shown) < len(frames):
        print(f"\n({len(shown)} of {len(frames)} shown)")


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------


def cmd_grid(args: argparse.Namespace) -> None:
    frames = load_frames(args.run)
    f = frame_at(frames, args.t)
    src = args.run / f["path"]
    if not src.exists():
        sys.exit(f"{src} not found.")

    img = Image.open(src).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    step = args.step
    for x in range(0, w + 1, step):
        major = x % (step * 5) == 0
        draw.line([(x, 0), (x, h)], fill=(255, 0, 0) if major else (0, 255, 255), width=1)
        if major:
            draw.text((x + 3, 3), str(x), fill=(255, 0, 0))
    for y in range(0, h + 1, step):
        major = y % (step * 5) == 0
        draw.line([(0, y), (w, y)], fill=(255, 0, 0) if major else (0, 255, 255), width=1)
        if major:
            draw.text((3, y + 3), str(y), fill=(255, 0, 0))

    # The horizon line, because a point defect above it cannot be ranged.
    hy = int(h * HORIZON_FRAC)
    draw.line([(0, hy), (w, hy)], fill=(255, 255, 0), width=2)
    draw.text((8, hy + 6), "approx horizon - point defects must sit BELOW", fill=(255, 255, 0))

    img.save(args.out, quality=92)
    print(f"t={f['t_sec']:.2f}s  frame {f['frame_id']}  {w}x{h}  ->  {args.out}")
    print("Read boxes off the grid as [x1, y1, x2, y2] in pixels.")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def lerp_box(a: list[float], b: list[float], u: float) -> list[float]:
    return [a[i] + (b[i] - a[i]) * u for i in range(4)]


def track_box(track: list[dict[str, Any]], t: float) -> list[float] | None:
    """Box at time `t`, interpolated between the marked keypoints.

    Outside the marked range this returns None rather than clamping: a defect is
    only visible for as long as you said it was, and extrapolating past the last
    mark would put a detection on a frame where the thing has gone.
    """
    if t < track[0]["t"] or t > track[-1]["t"]:
        return None
    for i in range(len(track) - 1):
        t0, t1 = track[i]["t"], track[i + 1]["t"]
        if t0 <= t <= t1:
            span = t1 - t0
            u = 0.0 if span <= 0 else (t - t0) / span
            return lerp_box(track[i]["box"], track[i + 1]["box"], u)
    return list(track[-1]["box"])


def cmd_build(args: argparse.Namespace) -> None:
    frames = load_frames(args.run)
    domain = load_domain(args.run)
    valid = set(domain.class_map)

    spec = yaml.safe_load(args.spec.read_text()) or {}
    defects = spec.get("defects") or []
    if not defects:
        sys.exit(f"{args.spec} has no `defects:` list.")

    truth: dict[str, list[dict[str, Any]]] = {}
    problems: list[str] = []
    total = 0

    for n, d in enumerate(defects, 1):
        who = f"defect #{n} ({d.get('cls', '?')})"

        cls = d.get("cls")
        if cls not in valid:
            problems.append(
                f"{who}: class {cls!r} is not in domain {domain.key!r}. "
                f"Known: {', '.join(sorted(valid))}"
            )
            continue

        track = sorted(d.get("track") or [], key=lambda k: k["t"])
        if not track:
            problems.append(f"{who}: no `track:` entries.")
            continue
        if any(len(k.get("box") or []) != 4 for k in track):
            problems.append(f"{who}: every track entry needs box [x1,y1,x2,y2].")
            continue

        pixels = str(d.get("units", "pixels")).lower() != "norm"
        severity = int(d.get("severity", 3))
        confidence = float(d.get("confidence", 0.9))
        evidence = str(d.get("evidence", "")).strip()
        if not evidence:
            problems.append(f"{who}: no `evidence:` text -- the panel will look empty.")

        spec_cls = domain.class_map[cls]
        hits = 0
        above_horizon = 0

        for f in frames:
            box = track_box(track, f["t_sec"])
            if box is None:
                continue

            if pixels:
                sx = BOX_SCALE / float(f["width"])
                sy = BOX_SCALE / float(f["height"])
                box = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]

            x1, y1, x2, y2 = (round(min(max(v, 0.0), BOX_SCALE), 1) for v in box)
            if x2 <= x1 or y2 <= y1:
                problems.append(f"{who}: degenerate box at t={f['t_sec']:.2f}s.")
                continue

            if spec_cls.geometry == "point" and y2 < BOX_SCALE * HORIZON_FRAC:
                above_horizon += 1

            truth.setdefault(f"{f['t_sec']:.2f}", []).append(
                {
                    "cls": cls,
                    "box": [x1, y1, x2, y2],
                    "severity": severity,
                    "confidence": confidence,
                    "evidence": evidence,
                }
            )
            hits += 1
            total += 1

        if above_horizon:
            problems.append(
                f"{who}: {above_horizon} box(es) sit above the horizon "
                f"(bottom edge in the top {HORIZON_FRAC:.0%} of the frame); "
                f"`localize` will not range a point class up there."
            )

        if hits == 0:
            problems.append(
                f"{who}: covers no keyframe. Its track spans "
                f"{track[0]['t']:.2f}s..{track[-1]['t']:.2f}s; the run has "
                f"{frames[0]['t_sec']:.2f}s..{frames[-1]['t_sec']:.2f}s."
            )
        elif hits == 1:
            problems.append(
                f"{who}: only 1 keyframe, so it gets a single ground-plane "
                f"projection (+/-2-4 m) rather than a triangulated position. "
                f"Widen the track to cover 3 or more keyframes."
            )
        print(f"{who}: {hits} keyframe(s)")

    args.out.write_text(json.dumps(truth, indent=1, sort_keys=True) + "\n")
    print(f"\n{total} detections over {len(truth)} keyframes -> {args.out}")

    if problems:
        print(f"\n{len(problems)} thing(s) to look at:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------


def cmd_preview(args: argparse.Namespace) -> None:
    frames = load_frames(args.run)
    domain = load_domain(args.run)
    truth = json.loads(args.truth.read_text())

    args.out.mkdir(parents=True, exist_ok=True)
    by_t = {f"{f['t_sec']:.2f}": f for f in frames}

    written = 0
    orphans = 0
    for key, items in sorted(truth.items()):
        if not items:
            continue
        f = by_t.get(key)
        if f is None:
            orphans += 1
            continue

        src = args.run / f["path"]
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        for it in items:
            x1 = it["box"][0] / BOX_SCALE * w
            y1 = it["box"][1] / BOX_SCALE * h
            x2 = it["box"][2] / BOX_SCALE * w
            y2 = it["box"][3] / BOX_SCALE * h

            spec = domain.class_map.get(it["cls"])
            colour = spec.color if spec else "#f59e0b"
            label = f"{spec.label if spec else it['cls']} - sev {it['severity']}"

            draw.rectangle([x1, y1, x2, y2], outline=colour, width=4)
            draw.rectangle(
                [x1, max(y1 - 22, 0), x1 + 8 * len(label), max(y1, 22)], fill=colour
            )
            draw.text((x1 + 4, max(y1 - 19, 3)), label, fill=(0, 0, 0))

        draw.text((10, 10), f"t={key}s  frame {f['frame_id']}", fill=(255, 255, 0))
        img.save(args.out / f"{key.replace('.', '_')}.jpg", quality=88)
        written += 1

    print(f"{written} preview frame(s) -> {args.out}")
    if orphans:
        print(
            f"WARNING: {orphans} truth key(s) match no keyframe in this run. The "
            f"mock backend looks fixtures up by exact timestamp, so those "
            f"detections would be silently dropped. Rebuild from the spec.",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("frames", help="List the run's keyframes.")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--from", dest="t_from", type=float, default=None)
    p.add_argument("--to", dest="t_to", type=float, default=None)
    p.add_argument("--every", type=int, default=1, help="Show every Nth keyframe.")
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("grid", help="Write one keyframe with a pixel grid.")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--t", type=float, required=True, help="Seconds; nearest keyframe wins.")
    p.add_argument("--out", type=Path, default=Path("grid.jpg"))
    p.add_argument("--step", type=int, default=50, help="Grid spacing in pixels.")
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("build", help="Spec (seconds + pixels) -> truth.json.")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("truth.json"))
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("preview", help="Draw a truth.json back onto the keyframes.")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("preview"))
    p.set_defaults(func=cmd_preview)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
