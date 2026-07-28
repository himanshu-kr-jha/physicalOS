#!/usr/bin/env python3
"""Convert a KITTI raw drive into PhysicalOS inputs: video + GPX + camera YAML.

WHY KITTI
It is the most readily available dataset shipping all three things this pipeline
needs, with the calibration as GROUND TRUTH rather than a guess:

  - image_02/data/*.png   forward-facing colour camera, 10 Hz
  - oxts/data/*.txt       OXTS RT3003 GPS/IMU, lat/lon per frame
  - calib_cam_to_cam.txt  exact rectified intrinsics

That makes it the real test of pos/geo.py: if localisation works here, the maths
is right on real imagery, not just on the synthetic sample.

LICENCE
KITTI is CC BY-NC-SA 3.0 -- attribution, NON-COMMERCIAL, share-alike. Fine for
validating the pipeline. Do NOT put KITTI frames in a customer or investor deck;
shoot your own footage for that.

DOWNLOAD (no account needed; the bucket is public)
    B=https://s3.eu-central-1.amazonaws.com/avg-kitti
    curl -O $B/raw_data/2011_09_26_calib.zip
    curl -C - -O $B/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
    unzip -q 2011_09_26_calib.zip && unzip -q 2011_09_26_drive_0002_sync.zip

Drives run 200 MB to several GB and the mirror is slow -- `curl -C -` resumes.

USAGE
    uv run python scripts/import_kitti.py \
        --drive 2011_09_26/2011_09_26_drive_0002_sync \
        --calib 2011_09_26 \
        --out samples/kitti

    uv run pos run --video samples/kitti/drive.mp4 \
        --gpx samples/kitti/track.gpx \
        --camera kitti --domain road --backend cosmos --out runs/kitti
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The KITTI sensor setup places the colour cameras 1.65 m above the road.
# Ground-plane projection needs this and KITTI does not ship it per drive.
KITTI_CAMERA_HEIGHT_M = 1.65

# oxts is logged at the image rate on *_sync drives.
KITTI_FPS = 10.0


def parse_calib(calib_dir: Path, cam: str = "02") -> dict:
    """Read rectified intrinsics for one camera from calib_cam_to_cam.txt."""
    path = calib_dir / "calib_cam_to_cam.txt"
    if not path.exists():
        raise SystemExit(
            f"No calib_cam_to_cam.txt in {calib_dir}.\n"
            "Download 2011_09_26_calib.zip and point --calib at the extracted "
            "date directory (e.g. 2011_09_26)."
        )

    fields: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.split()

    try:
        w, h = (int(float(x)) for x in fields[f"S_rect_{cam}"])
        p = [float(x) for x in fields[f"P_rect_{cam}"]]
    except KeyError as exc:
        raise SystemExit(f"Missing {exc} in {path}") from None

    # P_rect is 3x4 row-major: [fx 0 cx tx; 0 fy cy ty; 0 0 1 tz]
    fx, cx, fy, cy = p[0], p[2], p[5], p[6]

    return {
        "width": w,
        "height": h,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        # vfov comes from fy and the RECTIFIED height. KITTI frames are a wide
        # letterbox (1242x375), so vertical FOV is far narrower than a typical
        # dashcam: about 29 deg, not 58. Taking this from the calibration rather
        # than assuming a default is the entire point of this importer.
        "vfov_deg": 2 * math.degrees(math.atan((h / 2) / fy)),
        "hfov_deg": 2 * math.degrees(math.atan((w / 2) / fx)),
        # The principal point is not the image centre; that offset IS the pitch.
        "pitch_offset_frac": (cy - h / 2) / h,
    }


def parse_oxts(drive_dir: Path) -> tuple[list[tuple[float, float]], list[str]]:
    """Return ([(lat, lon), ...], [iso_timestamp, ...]) for the drive."""
    data_dir = drive_dir / "oxts" / "data"
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No oxts/data/*.txt under {drive_dir}")

    coords: list[tuple[float, float]] = []
    for f in files:
        parts = f.read_text().split()
        if len(parts) >= 2:
            coords.append((float(parts[0]), float(parts[1])))  # lat, lon

    stamps: list[str] = []
    ts_path = drive_dir / "oxts" / "timestamps.txt"
    if ts_path.exists():
        for line in ts_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # "2011-09-26 13:02:31.133278336" -- nanoseconds, wider than
            # fromisoformat accepts, so truncate to microseconds.
            head, _, frac = line.partition(".")
            micro = (frac + "000000")[:6]
            try:
                dt = datetime.strptime(f"{head}.{micro}", "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                continue
            # Milliseconds, not seconds: KITTI logs at 10 Hz, so second
            # resolution would collapse ten consecutive fixes onto one
            # timestamp and pos/ingest.py would interpolate a stepped path.
            stamps.append(
                dt.replace(tzinfo=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )

    return coords, stamps


def write_gpx(
    coords: list[tuple[float, float]], stamps: list[str], out: Path, fps: float
) -> None:
    """GPX track. Falls back to synthetic times if timestamps.txt is missing."""
    if len(stamps) < len(coords):
        base = datetime(2011, 9, 26, 13, 0, 0, tzinfo=timezone.utc)
        stamps = [
            (base + timedelta(seconds=i / fps))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
            for i in range(len(coords))
        ]

    pts = [
        f'      <trkpt lat="{lat:.8f}" lon="{lon:.8f}">'
        f"<time>{stamps[i]}</time></trkpt>"
        for i, (lat, lon) in enumerate(coords)
    ]
    out.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="PhysicalOS import_kitti" '
        'xmlns="http://www.topografix.com/GPX/1/1">\n'
        "  <trk>\n    <name>KITTI raw drive</name>\n    <trkseg>\n"
        + "\n".join(pts)
        + "\n    </trkseg>\n  </trk>\n</gpx>\n"
    )


def build_video(
    drive_dir: Path, out: Path, fps: float, cam: str = "02"
) -> tuple[int, int, int]:
    """Encode the PNG sequence into an mp4. Returns (n_frames, width, height).

    KITTI rectified frames are 1242x375 -- an ODD height, which libx264 refuses
    ("height not divisible by 2"). We pad ONE ROW AT THE BOTTOM rather than
    scaling or cropping, because padding at the bottom leaves every original
    pixel row at its original absolute index. That matters: the caller derives
    pitch_offset_frac from the principal point, and any change to row indices
    would silently move the horizon.

    The returned height is the ENCODED height, which is what pos/ingest.py will
    read back from the video, so the caller must compute the pitch fraction
    against it.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    frames_dir = drive_dir / f"image_{cam}" / "data"
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNGs under {frames_dir}")

    from PIL import Image

    with Image.open(pngs[0]) as im:
        src_w, src_h = im.size
    enc_w = src_w + (src_w % 2)
    enc_h = src_h + (src_h % 2)

    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps),
            # KITTI names frames 0000000000.png -- a 10-digit pattern.
            "-i", str(frames_dir / "%010d.png"),
            # Pad bottom-right to even dimensions; content stays at (0,0).
            "-vf", f"pad={enc_w}:{enc_h}:0:0:black",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{proc.stderr}")
    return len(pngs), enc_w, enc_h


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--drive", type=Path, required=True,
                   help="Extracted *_sync drive directory.")
    p.add_argument("--calib", type=Path, required=True,
                   help="Date directory holding calib_cam_to_cam.txt.")
    p.add_argument("--out", type=Path, default=Path("samples/kitti"))
    p.add_argument("--camera-name", default="kitti",
                   help="Written to configs/camera/<name>.yaml")
    p.add_argument("--height", type=float, default=KITTI_CAMERA_HEIGHT_M)
    p.add_argument("--fps", type=float, default=KITTI_FPS)
    p.add_argument("--cam", default="02", help="02 = left colour camera.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    calib = parse_calib(args.calib, args.cam)

    print("Calibration (from KITTI, exact -- not estimated)")
    print(f"  rectified size     : {calib['width']} x {calib['height']}")
    print(f"  fx, fy             : {calib['fx']:.2f}, {calib['fy']:.2f}")
    print(f"  principal point    : {calib['cx']:.2f}, {calib['cy']:.2f}")
    print(f"  vfov_deg           : {calib['vfov_deg']:.2f}")
    print(f"  pitch_offset_frac  : {calib['pitch_offset_frac']:+.4f}")
    print(f"  height_m           : {args.height} (KITTI sensor setup)")

    coords, stamps = parse_oxts(args.drive)
    gpx = args.out / "track.gpx"
    write_gpx(coords, stamps, gpx, args.fps)
    print(f"\nWrote {gpx}  ({len(coords)} track points)")

    video = args.out / "drive.mp4"
    n, enc_w, enc_h = build_video(args.drive, video, args.fps, args.cam)
    print(f"Wrote {video}  ({n} frames at {args.fps} fps, encoded {enc_w}x{enc_h})")

    # pitch_offset_frac is a FRACTION of frame height, and pos/ingest.py reads
    # the height back from the encoded video. If padding changed the height, the
    # fraction must be recomputed against it or the horizon shifts.
    pitch = (calib["cy"] - enc_h / 2) / enc_h
    if enc_h != calib["height"]:
        print(f"  padded {calib['height']} -> {enc_h} rows (libx264 needs even "
              f"dimensions); pitch recomputed {calib['pitch_offset_frac']:+.4f} "
              f"-> {pitch:+.4f}")

    cam_yaml = Path("configs/camera") / f"{args.camera_name}.yaml"
    cam_yaml.parent.mkdir(parents=True, exist_ok=True)
    cam_yaml.write_text(
        "# Derived from KITTI calib_cam_to_cam.txt by scripts/import_kitti.py\n"
        "# Exact intrinsics, not a guess. KITTI frames are a wide letterbox, so\n"
        "# the vertical FOV is much narrower than a normal dashcam.\n"
        f"# pitch computed against the ENCODED height ({enc_h} rows).\n"
        f"height_m: {args.height}\n"
        f"vfov_deg: {calib['vfov_deg']:.2f}\n"
        f"hfov_deg: {calib['hfov_deg']:.2f}\n"
        f"pitch_offset_frac: {pitch:.4f}\n"
        "max_range_m: 60.0\n"
    )
    print(f"Wrote {cam_yaml}")

    if coords:
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        print(f"\nRoute bbox: {min(lats):.6f},{min(lons):.6f} .. "
              f"{max(lats):.6f},{max(lons):.6f}")

    print("\nNow run:")
    print(f"  uv run pos run --video {video} --gpx {gpx} \\")
    print(f"      --camera {args.camera_name} --domain road \\")
    print(f"      --backend cosmos --out runs/{args.out.name}")
    print("\nReminder: KITTI is CC BY-NC-SA 3.0 -- non-commercial, attribution,")
    print("share-alike. Validate with it; do not ship its frames in a deck.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
