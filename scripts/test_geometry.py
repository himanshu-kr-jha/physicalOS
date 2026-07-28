#!/usr/bin/env python
"""Closed-form unit checks for the geometry primitives. Non-zero exit on failure.

WHY SEPARATE FROM verify_sample.py
verify_sample.py also has a geometry section, but it loads samples/road/objects.json
and exits 2 when the rendered sample is missing. A pure geometry check must not need
fixtures, or it cannot gate a pos/geo.py edit on a fresh clone. This file runs in
milliseconds with nothing but a camera config; verify_sample.py stays the
pipeline-output gate.

The checks that matter most are the REFUSALS. triangulate() must decline parallel
rays instead of returning a plausible number from an ill-conditioned solve: that
number would reach a user as "+/-0.1 m" and be believed. Refusing is the safe
failure; guessing is not.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pos.config import CameraConfig  # noqa: E402
from pos.geo import (  # noqa: E402
    bearing_to_box,
    box_ground_anchor,
    haversine_m,
    offset_latlon,
    ray_forward_distance,
    triangulate,
)

FAILED: list[str] = []


def chk(cond: bool, label: str, detail: str = "") -> None:
    if not cond:
        FAILED.append(label)
    print(
        f"  {'ok  ' if cond else 'FAIL'} {label}" + (f"  --  {detail}" if detail else "")
    )


def main() -> int:
    W, H = 1920, 1080
    cam = CameraConfig.load("dashcam")

    print("\n1. Triangulation recovers a known intersection")
    # Object 10 m north of A. B sits 6 m east of A, so it sees the object at
    # atan2(-6, 10) = -30.96 deg, stored as bearing 329.04.
    A = (25.0, 94.0)
    obj = offset_latlon(A[0], A[1], 10.0, 0.0)
    B = offset_latlon(A[0], A[1], 0.0, 6.0)
    brg_b = math.degrees(math.atan2(-6.0, 10.0)) % 360.0

    got = triangulate([(A[0], A[1], 0.0), (B[0], B[1], brg_b)])
    chk(got is not None, "two crossing rays produce a fix")
    if got:
        lat, lon, res, par = got
        err = haversine_m(lat, lon, obj[0], obj[1])
        chk(err < 0.10, "recovers the intersection to <0.1 m", f"{err * 100:.2f} cm")
        chk(res < 0.01, "residual ~0 for consistent rays", f"{res * 1000:.3f} mm")
        # Angular SEPARATION, not the raw bearing: 0 and 329.04 differ by 30.96.
        sep = abs((0.0 - brg_b + 180.0) % 360.0 - 180.0)
        chk(abs(par - sep) < 0.01, "parallax equals the true separation",
            f"{par:.2f} deg")

    print("\n2. Degenerate geometry is REFUSED, not fitted")
    par_pt = offset_latlon(A[0], A[1], 0.0, 5.0)
    chk(
        triangulate([(A[0], A[1], 0.0), (par_pt[0], par_pt[1], 0.0)]) is None,
        "exactly parallel rays refused",
    )
    chk(triangulate([(A[0], A[1], 45.0)]) is None, "a single ray refused")
    chk(triangulate([]) is None, "no rays refused")

    print("\n3. Inconsistent rays give a large residual (the precision filter)")
    bad = triangulate(
        [(A[0], A[1], 0.0), (B[0], B[1], brg_b), (B[0], B[1], brg_b + 25.0)]
    )
    chk(
        bad is not None and bad[2] > 1.0,
        "residual flags disagreement",
        f"{bad[2]:.2f} m" if bad else "no fix",
    )

    print("\n4. A fix behind the camera is detectable")
    back = offset_latlon(A[0], A[1], -10.0, 0.0)
    chk(
        ray_forward_distance(A[0], A[1], 0.0, back[0], back[1]) < 0,
        "behind the camera => negative forward distance",
    )
    fwd = ray_forward_distance(A[0], A[1], 0.0, obj[0], obj[1])
    chk(abs(fwd - 10.0) < 0.05, "ahead => correct distance", f"{fwd:.2f} m")

    print("\n5. Bearing depends on the box's horizontal centre only")
    mid = bearing_to_box([490, 400, 510, 500], W, H, cam, 90.0)
    chk(abs(mid - 90.0) < 0.01, "centred box bears exactly the heading", f"{mid:.3f}")
    left = bearing_to_box([0, 400, 20, 500], W, H, cam, 90.0)
    right = bearing_to_box([980, 400, 1000, 500], W, H, cam, 90.0)
    chk(left < 90.0 < right, "left bears less, right bears more",
        f"{left:.1f} / 90 / {right:.1f}")
    chk(abs((90.0 - left) - (right - 90.0)) < 0.01, "symmetric about the centre")
    # The entire reason triangulation beats ground projection.
    high = bearing_to_box([490, 0, 510, 10], W, H, cam, 90.0)
    chk(abs(high - mid) < 1e-9, "vertical position irrelevant (pitch-independent)")
    chk(
        0 <= bearing_to_box([490, 400, 510, 500], W, H, cam, 359.0) < 360,
        "bearing wraps into [0,360)",
    )

    print("\n6. Ground anchor depends on class geometry")
    box = [400.0, 200.0, 600.0, 800.0]
    _, v_point = box_ground_anchor(box, W, H, "point")
    _, v_area = box_ground_anchor(box, W, H, "area")
    _, v_default = box_ground_anchor(box, W, H)
    chk(v_point == v_default, "default unchanged (existing callers safe)")
    chk(abs(v_point - 0.800 * H) < 1e-6, "point anchors at the box bottom",
        f"v={v_point:.1f}")
    chk(abs(v_area - 0.500 * H) < 1e-6, "area anchors at the box centre",
        f"v={v_area:.1f}")
    chk(v_area < v_point, "area anchor sits higher => longer range")

    print("\n" + "=" * 64)
    if FAILED:
        print(f"  {len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("  All geometry checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
