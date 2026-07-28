#!/usr/bin/env python3
"""End-to-end correctness check against the synthetic sample.

The sample is built by placing objects at known world coordinates and
projecting them into pixels with an INDEPENDENT copy of the pinhole model (in
scripts/make_sample.py). The pipeline then inverts that projection using
pos/geo.py. So when the recovered lat/lon match the placed lat/lon, the
geometry in pos/geo.py is genuinely correct -- focal length, horizon, lateral
sign and heading rotation all agree.

That independence is the point. If pos/geo.py were used to build the fixtures
as well, this would prove nothing.

Usage:  uv run python scripts/verify_sample.py
Exit code is non-zero if any check fails, so it works as a CI gate.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pos.config import CameraConfig  # noqa: E402
from pos.geo import (  # noqa: E402
    box_ground_anchor,
    haversine_m,
    horizon_v,
    project_to_ground,
)

RUN = ROOT / "run"
SAMPLE = ROOT / "samples" / "road"

# A monocular ground-plane projection is good to a couple of metres, no better.
# The fixtures are exact, so on the synthetic sample we should beat this
# comfortably -- but the gate sits where a real-world claim would sit.
MAX_MEDIAN_ERR_M = 1.5
MAX_ERR_M = 3.0

failures: list[str] = []
notes: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def load(path: Path):
    if not path.exists():
        print(f"\nMissing {path}. Run the pipeline first:\n")
        print("  uv run python scripts/make_sample.py")
        print("  uv run pos run --video samples/road/road.mp4 \\")
        print("      --gpx samples/road/track.gpx \\")
        print("      --truth samples/road/truth.json --out run\n")
        sys.exit(2)
    return json.loads(path.read_text())


# --------------------------------------------------------------------------


def geometry_unit_checks(cam: CameraConfig) -> None:
    """pos/geo.py sanity, independent of any pipeline output."""
    print("\n1. Ground-plane projection unit checks")
    W, H = 1280, 720
    hz = horizon_v(H, cam)
    f_px = (H / 2) / math.tan(math.radians(cam.vfov_deg) / 2)

    # A row this far below the horizon lands at ~60% of max trusted range.
    mid_v = hz + cam.height_m * f_px / (0.6 * cam.max_range_m)

    near = project_to_ground(W / 2, H - 4, W, H, cam)
    mid = project_to_ground(W / 2, mid_v, W, H, cam)

    check(near is not None, "point near image bottom projects onto the ground")
    check(mid is not None, "point at mid-distance projects onto the ground")
    check(
        project_to_ground(W / 2, hz, W, H, cam) is None,
        "point AT the horizon has no ground range",
    )
    check(
        project_to_ground(W / 2, hz - 40, W, H, cam) is None,
        "point ABOVE the horizon has no ground range",
    )
    # Just below the horizon the range explodes (a few pixels = hundreds of
    # metres), which is exactly where a monocular estimate stops being usable.
    # Refusing to answer beyond max_range_m is the intended contract.
    check(
        project_to_ground(W / 2, hz + 2, W, H, cam) is None,
        f"range beyond max_range_m ({cam.max_range_m} m) is refused, not guessed",
    )

    if near and mid:
        check(
            near[0] < mid[0],
            "lower in frame means closer",
            f"bottom {near[0]:.1f} m < mid-frame {mid[0]:.1f} m",
        )

    left = project_to_ground(W * 0.25, H - 40, W, H, cam)
    right = project_to_ground(W * 0.75, H - 40, W, H, cam)
    if left and right:
        check(
            left[1] < 0 < right[1],
            "lateral sign flips across image centre",
            f"left {left[1]:+.2f} m, right {right[1]:+.2f} m",
        )

    # Analytic cross-check of the focal-length maths.
    f = (H / 2) / math.tan(math.radians(cam.vfov_deg) / 2)
    v = hz + 200.0
    expect = cam.height_m / math.tan(math.atan2(v - hz, f))
    got = project_to_ground(W / 2, v, W, H, cam)
    if got:
        check(
            abs(got[0] - expect) < 1e-6,
            "forward range matches the closed-form solution",
            f"{got[0]:.4f} m vs {expect:.4f} m",
        )

    # The anchor must be BOTTOM-centre, in a 0..1000 top-left frame.
    u, v2 = box_ground_anchor([400.0, 200.0, 600.0, 800.0], W, H)
    check(
        abs(u - 0.5 * W) < 1e-6 and abs(v2 - 0.8 * H) < 1e-6,
        "box anchor is bottom-centre in 0-1000 top-left coords",
        f"u={u:.1f} (want {0.5 * W:.0f}), v={v2:.1f} (want {0.8 * H:.0f})",
    )


def frame_checks(frames: list[dict]) -> None:
    print("\n2. Ingest: keyframes are georeferenced")
    check(len(frames) > 0, "keyframes were extracted", f"{len(frames)} frames")
    check(
        all(0 <= f["heading_deg"] < 360 for f in frames),
        "every heading is in [0, 360)",
    )
    check(
        all(f["width"] > 0 and f["height"] > 0 for f in frames),
        "every frame has real pixel dimensions",
        f"{frames[0]['width']}x{frames[0]['height']}",
    )

    ts = [f["t_sec"] for f in frames]
    check(
        all(b > a for a, b in zip(ts, ts[1:])),
        "timestamps increase strictly",
        f"{ts[0]:.2f}s .. {ts[-1]:.2f}s",
    )

    # The sample drives due east, so heading should sit near 90 degrees.
    mid = frames[len(frames) // 2]["heading_deg"]
    check(abs(mid - 90.0) < 5.0, "heading matches the eastbound route", f"{mid:.1f} deg")

    lats = [f["lat"] for f in frames]
    lons = [f["lon"] for f in frames]
    notes.append(
        f"route spans lat {min(lats):.6f}..{max(lats):.6f}, "
        f"lon {min(lons):.6f}..{max(lons):.6f}"
    )


def dedupe_checks(manifest: dict, findings: list[dict], objects: list[dict]) -> None:
    print("\n3. Clustering: repeated sightings collapse to one finding each")
    n_det = manifest.get("n_detections", 0)

    check(
        len(findings) == len(objects),
        "finding count equals the number of real world objects",
        f"{n_det} detections -> {len(findings)} findings, {len(objects)} objects placed",
    )
    check(
        n_det >= len(findings),
        "there were more detections than findings (merging happened)",
        f"{n_det} -> {len(findings)}",
    )

    multi = [f for f in findings if len(f["evidence"]) > 1]
    check(
        len(multi) > 0,
        "at least one finding carries several sightings",
        f"{len(multi)}/{len(findings)} findings have 2+ evidence frames",
    )
    check(
        all(len(f["evidence"]) >= 1 for f in findings),
        "every finding is backed by at least one frame",
    )

    ids = [f["finding_id"] for f in findings]
    check(len(ids) == len(set(ids)), "finding ids are unique")


def position_checks(findings: list[dict], objects: list[dict]) -> None:
    print("\n4. Localisation: recovered positions match placed positions")

    errs: list[tuple[float, str]] = []
    used: set[int] = set()
    missed: list[str] = []

    for obj in objects:
        best, best_d = -1, float("inf")
        for i, f in enumerate(findings):
            if i in used or f["cls"] != obj["cls"] or f["lat"] is None:
                continue
            d = haversine_m(obj["lat"], obj["lon"], f["lat"], f["lon"])
            if d < best_d:
                best, best_d = i, d
        if best < 0:
            missed.append(obj["cls"])
            continue
        used.add(best)
        errs.append((best_d, obj["cls"]))

    check(not missed, "every placed object was recovered", f"missed: {missed or 'none'}")
    if not errs:
        check(False, "position errors could be computed")
        return

    errs.sort()
    median = errs[len(errs) // 2][0]
    p90 = errs[min(int(len(errs) * 0.9), len(errs) - 1)][0]
    worst, worst_cls = errs[-1]

    check(
        median <= MAX_MEDIAN_ERR_M,
        f"median position error within {MAX_MEDIAN_ERR_M} m",
        f"{median:.2f} m",
    )
    check(
        worst <= MAX_ERR_M,
        f"worst position error within {MAX_ERR_M} m",
        f"{worst:.2f} m ({worst_cls})",
    )
    notes.append(
        f"position error: median {median:.2f} m, p90 {p90:.2f} m, max {worst:.2f} m"
    )


def evidence_checks(findings: list[dict], frames: list[dict]) -> None:
    print("\n5. Evidence: boxes are in range and land on real pixels")
    by_id = {f["frame_id"]: f for f in frames}

    bad_range = missing_img = 0
    for f in findings:
        for d in f["evidence"]:
            x1, y1, x2, y2 = d["box"]
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                bad_range += 1
            fr = by_id.get(d["frame_id"])
            if fr is None or not (RUN / fr["path"]).exists():
                missing_img += 1

    check(
        bad_range == 0,
        "all boxes lie within 0-1000 and are non-degenerate",
        f"{bad_range} bad",
    )
    check(
        missing_img == 0,
        "every evidence frame image exists on disk",
        f"{missing_img} missing",
    )

    # Contrast test: rendered defects are darker or lighter than the road, so a
    # correctly placed box should differ from its surroundings. This is what
    # catches a box that is numerically valid but sitting on blank asphalt --
    # exactly the failure mode a frame-id/timestamp mismatch produces.
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        notes.append("skipped pixel-contrast check (numpy/Pillow unavailable)")
        return

    checked = weak = 0
    for f in findings:
        if f["cls"] == "streetlight":
            continue  # mostly above the horizon; its box covers sky
        d = max(
            f["evidence"],
            key=lambda e: (e["box"][2] - e["box"][0]) * (e["box"][3] - e["box"][1]),
        )
        fr = by_id.get(d["frame_id"])
        if fr is None:
            continue

        img = Image.open(RUN / fr["path"]).convert("RGB")
        W, H = img.size
        x1 = d["box"][0] / 1000 * W
        y1 = d["box"][1] / 1000 * H
        x2 = d["box"][2] / 1000 * W
        y2 = d["box"][3] / 1000 * H
        if (x2 - x1) < 8 or (y2 - y1) < 6:
            continue

        inner = np.asarray(img.crop((int(x1), int(y1), int(x2), int(y2))), dtype=float)
        pad = max(12, int((x2 - x1) * 0.8))
        outer = np.asarray(
            img.crop(
                (
                    int(max(0, x1 - pad)),
                    int(max(0, y1 - pad)),
                    int(min(W, x2 + pad)),
                    int(min(H, y2 + pad)),
                )
            ),
            dtype=float,
        )
        checked += 1
        if abs(inner.mean() - outer.mean()) < 2.0:
            weak += 1

    if checked:
        check(
            (checked - weak) / checked >= 0.75,
            "closest box of each finding differs from its surroundings",
            f"{checked - weak}/{checked} boxes show contrast",
        )


# --------------------------------------------------------------------------


def main() -> int:
    print("PhysicalOS sample verification")
    print("=" * 64)

    cam = CameraConfig.load("dashcam")
    frames = load(RUN / "frames.json")
    findings = load(RUN / "findings.json")
    manifest = load(RUN / "manifest.json")
    objects = load(SAMPLE / "objects.json")

    geometry_unit_checks(cam)
    frame_checks(frames)
    dedupe_checks(manifest, findings, objects)
    position_checks(findings, objects)
    evidence_checks(findings, frames)

    print("\n" + "=" * 64)
    for n in notes:
        print(f"  note: {n}")

    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
