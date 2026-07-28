#!/usr/bin/env python3
"""Solve camera vfov and pitch from two ground measurements.

WHY THIS EXISTS
`height_m` you can measure with a tape. The other two numbers you cannot:

  - vfov_deg          nobody publishes the VERTICAL field of view, and video
                      often crops differently from stills, so the spec sheet
                      misleads even when you find it.
  - pitch_offset_frac depends on exactly how the mount sits today.

Both scale every distance the pipeline reports, so guessing them is the largest
avoidable error in the system. But two unknowns need only two measurements, and
the solution is closed-form.

THE PROCEDURE (5 minutes, needs a tape measure)
  1. Park. Leave the camera exactly as you drive with it -- do not adjust it.
  2. Put a marker on the ground straight ahead at a measured distance, near:
     say 5 m. A water bottle, a chalk cross, a shoe.
  3. Put a second marker further out: say 15 m. Further apart is better.
  4. Record a few seconds, or grab a still.
  5. Open the frame in any image viewer and read the PIXEL ROW where each
     marker meets the ground -- its base, not its top. GIMP, Preview and even
     Paint show cursor coordinates.
  6. Run this script with those two pairs.

  uv run python scripts/calibrate.py \
      --height 1.35 --image-height 720 --near 5:612 --far 15:451

THE MATHS
For a ground point at forward distance Z the pixel row is

    v = cy + f*h/Z          where f = (H/2)/tan(vfov/2),  cy = H/2 + pitch*H

Two measurements (Z1,v1) and (Z2,v2) give two equations. Subtracting cancels
cy, so f falls straight out:

    f  = (v1 - v2) / ( h * (1/Z1 - 1/Z2) )
    cy = v1 - f*h/Z1

then vfov = 2*atan((H/2)/f) and pitch = (cy - H/2)/H.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pos.config import CameraConfig  # noqa: E402
from pos.geo import project_to_ground  # noqa: E402


def parse_pair(text: str, label: str) -> tuple[float, float]:
    """Parse 'distance_m:pixel_row'."""
    if ":" not in text:
        raise SystemExit(f"--{label} must look like METRES:PIXELROW, e.g. 5:612")
    d, v = text.split(":", 1)
    try:
        return float(d), float(v)
    except ValueError:
        raise SystemExit(f"--{label}: could not parse {text!r}") from None


def solve(
    height_m: float,
    image_h: int,
    near: tuple[float, float],
    far: tuple[float, float],
) -> tuple[float, float]:
    """Return (vfov_deg, pitch_offset_frac)."""
    z1, v1 = near
    z2, v2 = far

    if z1 <= 0 or z2 <= 0:
        raise SystemExit("Distances must be positive.")
    if abs(z1 - z2) < 0.5:
        raise SystemExit("The two markers are too close together to solve reliably.")
    if v1 <= v2:
        raise SystemExit(
            f"The nearer marker ({z1} m) must sit LOWER in the frame -- a larger "
            f"pixel row -- than the far one ({z2} m).\n"
            f"Got near row {v1} and far row {v2}. Check you have not swapped them, "
            "and remember row 0 is the TOP of the image."
        )

    denom = height_m * (1.0 / z1 - 1.0 / z2)
    if abs(denom) < 1e-9:
        raise SystemExit("Degenerate measurement; move the markers further apart.")

    f = (v1 - v2) / denom
    if f <= 0:
        raise SystemExit("Solved a negative focal length; check your inputs.")

    cy = v1 - f * height_m / z1
    vfov = 2.0 * math.degrees(math.atan((image_h / 2.0) / f))
    pitch = (cy - image_h / 2.0) / image_h
    return vfov, pitch


def main() -> int:
    p = argparse.ArgumentParser(
        description="Solve vfov_deg and pitch_offset_frac from two ground markers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--height", type=float, required=True,
                   help="Camera height above ground in metres (measure it).")
    p.add_argument("--image-height", type=int,
                   help="Frame height in pixels. Inferred from --frame if given.")
    p.add_argument("--frame", type=Path,
                   help="A frame image, used only to read its dimensions.")
    p.add_argument("--near", required=True, metavar="M:ROW",
                   help="Nearer marker, e.g. 5:612")
    p.add_argument("--far", required=True, metavar="M:ROW",
                   help="Further marker, e.g. 15:451")
    p.add_argument("--max-range", type=float, default=60.0)
    p.add_argument("--write", type=Path,
                   help="Write a camera YAML here, e.g. configs/camera/mycam.yaml")
    args = p.parse_args()

    image_h = args.image_height
    image_w = None
    if args.frame:
        from PIL import Image

        with Image.open(args.frame) as im:
            image_w, ih = im.size
        image_h = image_h or ih
    if not image_h:
        raise SystemExit("Provide --image-height or --frame.")

    near = parse_pair(args.near, "near")
    far = parse_pair(args.far, "far")
    vfov, pitch = solve(args.height, image_h, near, far)

    print("Solved calibration")
    print("=" * 52)
    print(f"  height_m:           {args.height}")
    print(f"  vfov_deg:           {vfov:.2f}")
    print(f"  pitch_offset_frac:  {pitch:+.4f}")
    print(f"  -> horizon sits at pixel row "
          f"{image_h / 2.0 + pitch * image_h:.0f} of {image_h}")

    if not (20.0 < vfov < 140.0):
        print(f"\n  WARNING: {vfov:.1f} deg is outside the plausible range for a")
        print("  normal lens. Re-check your two pixel rows and distances.")
    if abs(pitch) > 0.35:
        print(f"\n  WARNING: pitch {pitch:+.2f} puts the horizon near the frame edge.")
        print("  That is unusual; re-check your measurements.")

    # Round-trip through the real projection code: feed the measured rows back
    # in and confirm we recover the distances we started from. This catches any
    # inconsistency between this solver and pos/geo.py.
    cam = CameraConfig(
        height_m=args.height,
        vfov_deg=vfov,
        pitch_offset_frac=pitch,
        max_range_m=max(args.max_range, far[0] * 2),
    )
    print("\nVerification (re-projecting your own measurements):")
    ok = True
    for z, v in (near, far):
        width = image_w or 1280
        got = project_to_ground(width / 2, v, width, image_h, cam)
        if got is None:
            print(f"  row {v:.0f}: FAILED to project")
            ok = False
            continue
        err = abs(got[0] - z)
        print(f"  row {v:.0f} -> {got[0]:6.2f} m  (measured {z:.2f} m)  "
              f"{'ok' if err < 0.05 else 'MISMATCH'}")
        ok = ok and err < 0.05
    print("  consistent with pos/geo.py" if ok else "  INCONSISTENT -- please report")

    yaml_text = (
        "# Solved by scripts/calibrate.py\n"
        f"height_m: {args.height}\n"
        f"vfov_deg: {vfov:.2f}\n"
        f"hfov_deg: {vfov * 16 / 9:.2f}   # rough 16:9 estimate; unused by the maths\n"
        f"pitch_offset_frac: {pitch:.4f}\n"
        f"max_range_m: {args.max_range}\n"
    )

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(yaml_text)
        print(f"\nWrote {args.write}")
        print(f"Use it with:  --camera {args.write.stem}")
    else:
        print("\nPaste into configs/camera/<name>.yaml:\n")
        print(yaml_text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
