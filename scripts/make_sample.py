#!/usr/bin/env python3
"""Render a synthetic dashcam drive with defects at KNOWN world positions.

Why this exists: a demo that needs someone's private video and a paid API key
before it shows anything is not a demo. This produces a complete, committed
sample -- video, GPS track, and ground truth -- so `pos run` works on a fresh
clone with no key and no GPU.

It is also a real correctness check. Defects are placed in world coordinates
(metres along the road, metres lateral) and projected into pixels using the
SAME pinhole model as pos/geo.py. So when the pipeline later projects those
pixels back onto the ground plane, it should recover the original world
positions. A sign flip or focal-length error in pos/geo.py shows up as visibly
wrong recovered positions.

Usage:  uv run python scripts/make_sample.py
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "road"

W, H = 1280, 720
RENDER_FPS = 10
SAMPLE_FPS = 2.0          # what the pipeline extracts; truth is keyed to this
DURATION_S = 30.0
SPEED_MPS = 8.0           # ~29 km/h

# Must match configs/camera/dashcam.yaml
CAM_HEIGHT_M = 1.35
VFOV_DEG = 58.0

# Route: a straight road heading due east from this origin.
ORIGIN_LAT, ORIGIN_LON = 12.971600, 77.594600
START_TIME = datetime(2026, 7, 25, 9, 14, 0, tzinfo=timezone.utc)

ROAD_HALF_W = 3.5         # metres from centreline to kerb
FOOTPATH_W = 2.0

F_PX = (H / 2.0) / math.tan(math.radians(VFOV_DEG) / 2.0)
CX, CY = W / 2.0, H / 2.0

SKY_TOP = (108, 152, 195)
SKY_BOT = (196, 214, 230)
ROAD = (78, 78, 82)
ROAD_EDGE = (58, 58, 62)
VERGE = (104, 112, 88)
FOOTPATH = (156, 152, 146)
KERB = (188, 186, 182)
BUILDING = (128, 124, 128)
BUILDING_DARK = (104, 100, 104)


class Obj:
    """A thing in the world. `s` = metres along road, `x` = metres lateral."""

    def __init__(
        self,
        cls: str,
        s: float,
        x: float,
        size: tuple[float, float],
        severity: int,
        evidence: str,
        height_m: float = 0.0,
    ):
        self.cls = cls
        self.s = s
        self.x = x
        self.w_m, self.l_m = size       # width (lateral), length (along road)
        self.severity = severity
        self.evidence = evidence
        self.height_m = height_m        # 0 = flat on the ground


# Laid out as four stretches of deliberately different quality, so the heatmap
# has something real to show: sound at the start, badly failed in the middle,
# recovering, then sound again. A uniformly bad road makes a boring heatmap and
# a uniformly good one makes a pointless demo.
OBJECTS: list[Obj] = [
    # --- Stretch 1 (s 12-42): sound carriageway, isolated minor defects -----
    Obj("crack", 22, 0.3, (1.8, 0.3), 1,
        "Hairline longitudinal crack along the lane line, no displacement."),
    Obj("patch_repair", 34, -1.0, (1.6, 1.4), 1,
        "Neat asphalt patch over a service trench, flush with the surface."),
    Obj("streetlight", 20, 5.6, (0.25, 0.25), 1,
        "Street lighting pole with luminaire head, appears intact.", height_m=7.0),

    # --- Stretch 2 (s 48-118): severe failure, the red zone ----------------
    Obj("pothole", 50, -1.2, (0.8, 0.7), 4,
        "Circular cavity in the wheel path with broken, spalled edges and visible depth."),
    Obj("pothole", 55, 1.5, (1.0, 0.9), 5,
        "Large deep pothole across the offside wheel path, edges fully broken away."),
    Obj("crack", 59, 0.2, (2.4, 0.4), 3,
        "Interconnected alligator cracking indicating fatigue failure of the base."),
    Obj("pothole", 63, -0.5, (0.7, 0.6), 4,
        "Pothole at the lane centre with loose aggregate scattered around the rim."),
    Obj("waterlogging", 69, -1.8, (2.6, 2.0), 4,
        "Standing water ponding against the kerb, covering the near wheel path."),
    Obj("pothole", 75, 1.2, (0.9, 0.85), 5,
        "Deep pothole with undercut edges, exposed base layer visible."),
    Obj("hazard", 80, 0.6, (0.85, 0.85), 5,
        "Open manhole with no cover and no barricade, directly in the running lane."),
    Obj("crack", 85, -0.4, (2.6, 0.45), 4,
        "Severe block cracking across the near lane with vertical displacement."),
    Obj("pothole", 89, 1.6, (0.8, 0.7), 4,
        "Pothole forming at the edge of a failed reinstatement."),
    Obj("garbage", 93, 4.5, (1.2, 1.0), 3,
        "Uncontained refuse dumped across the footpath and kerb line."),
    Obj("footpath_damaged", 97, 4.6, (1.5, 1.3), 4,
        "Paving slabs subsided and fractured, a clear trip hazard for pedestrians."),
    Obj("pothole", 102, -1.0, (0.7, 0.6), 3,
        "Pothole developing at a longitudinal joint."),
    Obj("waterlogging", 107, 1.4, (2.2, 1.7), 3,
        "Ponded water over the offside lane with no visible drainage."),
    Obj("footpath_obstruction", 111, 4.8, (1.3, 1.1), 3,
        "Material stacked across the footpath, forcing pedestrians into the carriageway."),
    Obj("pothole", 116, 0.9, (0.85, 0.75), 4,
        "Pothole at the lane divide with broken, ravelling edges."),
    Obj("streetlight", 60, 5.6, (0.25, 0.25), 1,
        "Street lighting pole with luminaire head, appears intact.", height_m=7.0),

    # --- Stretch 3 (s 126-166): moderate wear ------------------------------
    Obj("crack", 128, 0.3, (2.0, 0.35), 2,
        "Longitudinal cracking following the lane line, minor ravelling."),
    Obj("patch_repair", 137, -1.1, (1.8, 1.5), 1,
        "Asphalt patch over a utility trench, surface slightly proud of the road."),
    Obj("footpath_damaged", 145, 4.5, (1.3, 1.1), 2,
        "Two cracked paving slabs with a small level difference."),
    Obj("pothole", 158, 1.1, (0.65, 0.6), 3,
        "Small pothole at the edge of a previous repair."),
    Obj("streetlight", 150, 5.6, (0.25, 0.25), 1,
        "Street lighting pole with luminaire head, appears intact.", height_m=7.0),

    # --- Stretch 4 (s 174-226): sound again --------------------------------
    Obj("crack", 176, -0.2, (1.6, 0.3), 1,
        "Fine surface crazing, cosmetic only."),
    Obj("garbage", 198, 4.4, (1.0, 0.9), 2,
        "Small pile of dumped waste on the footpath verge."),
    Obj("patch_repair", 212, 0.8, (1.7, 1.4), 1,
        "Well-finished patch repair, no surface irregularity."),
    Obj("footpath_damaged", 224, 4.7, (1.2, 1.0), 2,
        "Single fractured paving slab on an otherwise sound footpath."),
    Obj("streetlight", 190, 5.6, (0.25, 0.25), 1,
        "Street lighting pole with luminaire head, appears intact.", height_m=7.0),
]

# Visual context: (s_start, s_end, lateral, height)
BUILDINGS = [
    (10, 40, 12.0, 11.0), (46, 78, 12.5, 15.0), (84, 118, 12.0, 9.0),
    (126, 160, 13.0, 18.0), (166, 205, 12.0, 12.0),
    (14, 44, -12.0, 13.0), (52, 86, -12.5, 10.0), (94, 130, -12.0, 16.0),
    (140, 180, -13.0, 11.0),
]


# ---------------------------------------------------------------------------
# Projection -- mirrors pos/geo.py exactly
# ---------------------------------------------------------------------------


def project(forward_m: float, lateral_m: float, height_m: float = 0.0):
    """World (relative to camera) -> pixel. None if behind or at the horizon."""
    if forward_m <= 0.6:
        return None
    u = CX + F_PX * lateral_m / forward_m
    v = CY + F_PX * (CAM_HEIGHT_M - height_m) / forward_m
    return u, v


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def draw_sky(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    hy = int(CY)
    for y in range(0, hy + 1):
        f = y / max(hy, 1)
        d.line(
            [(0, y), (W, y)],
            fill=(
                int(SKY_TOP[0] + (SKY_BOT[0] - SKY_TOP[0]) * f),
                int(SKY_TOP[1] + (SKY_BOT[1] - SKY_TOP[1]) * f),
                int(SKY_TOP[2] + (SKY_BOT[2] - SKY_TOP[2]) * f),
            ),
        )
    d.rectangle([0, hy, W, H], fill=VERGE)


def draw_surface(d: ImageDraw.ImageDraw, cam_s: float) -> None:
    """Road, kerb and footpath as a stack of depth slices.

    Slicing by depth rather than drawing one big trapezoid keeps the
    perspective exact and lets each strip have its own width, which is what
    makes the footpath and kerb read correctly.
    """
    far, near = 90.0, 0.8
    steps = 150
    for i in range(steps):
        z0 = far * (1 - i / steps) ** 2 + near
        z1 = far * (1 - (i + 1) / steps) ** 2 + near
        if z1 >= z0:
            continue

        def strip(x_left: float, x_right: float, color):
            a = project(z0, x_left)
            b = project(z0, x_right)
            c = project(z1, x_right)
            e = project(z1, x_left)
            if None in (a, b, c, e):
                return
            d.polygon([a, b, c, e], fill=color)

        strip(-ROAD_HALF_W - 0.35, ROAD_HALF_W + 0.35, ROAD_EDGE)
        strip(-ROAD_HALF_W, ROAD_HALF_W, ROAD)
        # Footpath on the right (positive lateral), with a kerb face.
        strip(ROAD_HALF_W + 0.35, ROAD_HALF_W + 0.6, KERB)
        strip(ROAD_HALF_W + 0.6, ROAD_HALF_W + 0.6 + FOOTPATH_W, FOOTPATH)

    # Centre dashes: 3 m mark, 6 m gap, positioned in world space so they slide
    # past the camera at the correct rate.
    period = 9.0
    first = math.floor(cam_s / period) * period
    for k in range(-1, 12):
        s0 = first + k * period
        z0, z1 = s0 - cam_s, s0 + 3.0 - cam_s
        if z1 <= 1.0:
            continue
        quad = [
            project(max(z0, 1.0), -0.09), project(max(z0, 1.0), 0.09),
            project(z1, 0.09), project(z1, -0.09),
        ]
        if None not in quad:
            d.polygon(quad, fill=(214, 210, 196))

    # Edge lines.
    for side in (-1, 1):
        for k in range(0, 88, 2):
            z0, z1 = float(k) + 1.0, float(k) + 2.6
            quad = [
                project(z0, side * (ROAD_HALF_W - 0.22)),
                project(z0, side * (ROAD_HALF_W - 0.10)),
                project(z1, side * (ROAD_HALF_W - 0.10)),
                project(z1, side * (ROAD_HALF_W - 0.22)),
            ]
            if None not in quad:
                d.polygon(quad, fill=(196, 192, 178))


def draw_buildings(d: ImageDraw.ImageDraw, cam_s: float) -> None:
    # Far to near, so nearer buildings occlude correctly.
    for s0, s1, lat, height in sorted(BUILDINGS, key=lambda b: -(b[0] - cam_s)):
        z0, z1 = s0 - cam_s, s1 - cam_s
        if z1 <= 2.0:
            continue
        z0 = max(z0, 2.0)

        base_near = project(z0, lat)
        base_far = project(z1, lat)
        top_near = project(z0, lat, height)
        top_far = project(z1, lat, height)
        if None in (base_near, base_far, top_near, top_far):
            continue

        d.polygon(
            [base_near, base_far, top_far, top_near],
            fill=BUILDING if lat > 0 else BUILDING_DARK,
            outline=(72, 70, 74),
        )
        # Window rows, purely so the street reads as a street.
        rows = max(1, int(height // 3.2))
        for r in range(rows):
            wy = 1.2 + r * 3.2
            if wy + 1.1 > height:
                break
            a = project(z0, lat, wy)
            b = project(z1, lat, wy)
            c = project(z1, lat, wy + 1.1)
            e = project(z0, lat, wy + 1.1)
            if None not in (a, b, c, e):
                d.polygon([a, b, c, e], fill=(88, 96, 112))


def _obj_bbox(obj: Obj, cam_s: float):
    """Pixel bbox of an object, or None when off-screen or too far."""
    z = obj.s - cam_s
    if z <= 1.5 or z > 55.0:
        return None

    if obj.height_m > 0:
        # Vertical object: the box spans base to top.
        base = project(z, obj.x)
        top = project(z, obj.x, obj.height_m)
        if base is None or top is None:
            return None
        half_w_px = max(3.0, F_PX * (obj.w_m / 2.0) / z)
        return (base[0] - half_w_px, top[1], base[0] + half_w_px, base[1])

    # Flat object: project its four ground corners.
    corners = [
        project(z - obj.l_m / 2, obj.x - obj.w_m / 2),
        project(z - obj.l_m / 2, obj.x + obj.w_m / 2),
        project(z + obj.l_m / 2, obj.x + obj.w_m / 2),
        project(z + obj.l_m / 2, obj.x - obj.w_m / 2),
    ]
    if any(c is None for c in corners):
        return None
    xs = [c[0] for c in corners]  # type: ignore[index]
    ys = [c[1] for c in corners]  # type: ignore[index]
    return (min(xs), min(ys), max(xs), max(ys))


def draw_objects(d: ImageDraw.ImageDraw, cam_s: float) -> None:
    for obj in sorted(OBJECTS, key=lambda o: -(o.s - cam_s)):
        bb = _obj_bbox(obj, cam_s)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        w, h = x2 - x1, y2 - y1

        if obj.cls == "streetlight":
            d.rectangle([x1, y1, x2, y2], fill=(92, 92, 96))
            head = max(3.0, w * 3.2)
            cx = (x1 + x2) / 2
            d.ellipse([cx - head, y1 - head * 0.5, cx + head, y1 + head * 0.5],
                      fill=(232, 226, 180))
        elif obj.cls == "pothole":
            d.ellipse([x1, y1, x2, y2], fill=(28, 26, 26))
            d.ellipse([x1 + w * 0.18, y1 + h * 0.2, x2 - w * 0.18, y2 - h * 0.2],
                      fill=(16, 14, 14))
        elif obj.cls == "crack":
            pts = []
            steps = 9
            for i in range(steps + 1):
                t = i / steps
                pts.append((x1 + w * t, y1 + h * (0.5 + 0.42 * math.sin(t * 7.0))))
            d.line(pts, fill=(30, 28, 28), width=max(2, int(h * 0.18)))
        elif obj.cls == "waterlogging":
            d.ellipse([x1, y1, x2, y2], fill=(96, 118, 132))
            d.ellipse([x1 + w * 0.25, y1 + h * 0.3, x2 - w * 0.3, y2 - h * 0.25],
                      fill=(126, 148, 160))
        elif obj.cls == "footpath_damaged":
            d.rectangle([x1, y1, x2, y2], fill=(122, 118, 112))
            d.line([(x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2)], fill=(70, 68, 66), width=2)
            d.line([((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2)], fill=(70, 68, 66), width=2)
        elif obj.cls == "garbage":
            d.ellipse([x1, y1, x2, y2], fill=(112, 104, 76))
            d.ellipse([x1, y1 + h * 0.35, (x1 + x2) / 2, y2], fill=(138, 128, 92))
        elif obj.cls == "hazard":
            d.ellipse([x1, y1, x2, y2], fill=(12, 12, 14))
            d.arc([x1, y1, x2, y2], 0, 360, fill=(168, 150, 60), width=3)
        elif obj.cls == "patch_repair":
            d.rectangle([x1, y1, x2, y2], fill=(52, 50, 52), outline=(38, 36, 38))
        elif obj.cls == "footpath_obstruction":
            d.rectangle([x1, y1, x2, y2], fill=(148, 116, 72), outline=(96, 74, 46))


def render_frame(cam_s: float) -> Image.Image:
    img = Image.new("RGB", (W, H), SKY_BOT)
    draw_sky(img)
    d = ImageDraw.Draw(img)
    draw_buildings(d, cam_s)
    draw_surface(d, cam_s)
    draw_objects(d, cam_s)
    return img


# ---------------------------------------------------------------------------
# Truth, GPX
# ---------------------------------------------------------------------------


def build_truth() -> dict[str, list[dict]]:
    """Truth keyed by TIMESTAMP, for every rendered frame.

    Keying by frame_id would be wrong: which source frame ends up as output
    frame 00014 depends on ffmpeg's sampling, so the fixture would sit on the
    wrong pixels. Emitting an entry for every rendered frame, keyed by its
    exact time, means the lookup lines up no matter what rate the pipeline
    samples at -- 2 fps, 5 fps, or every frame.
    """
    truth: dict[str, list[dict]] = {}

    for i in range(int(DURATION_S * RENDER_FPS)):
        t = i / RENDER_FPS
        cam_s = t * SPEED_MPS
        items: list[dict] = []

        for obj in OBJECTS:
            bb = _obj_bbox(obj, cam_s)
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            # Ignore anything too small to be a credible detection. A flat
            # defect 30 m away really is only a few pixels tall, so this gate
            # is what keeps the fixtures honest about detection range.
            if (x2 - x1) < 10 or (y2 - y1) < 6:
                continue
            if x2 < 0 or x1 > W or y2 < 0 or y1 > H:
                continue

            items.append(
                {
                    "cls": obj.cls,
                    "box": [
                        round(max(0.0, x1) / W * 1000, 1),
                        round(max(0.0, y1) / H * 1000, 1),
                        round(min(float(W), x2) / W * 1000, 1),
                        round(min(float(H), y2) / H * 1000, 1),
                    ],
                    "severity": obj.severity,
                    "confidence": 0.92,
                    "evidence": obj.evidence,
                }
            )
        truth[f"{t:.2f}"] = items
    return truth


def build_gpx() -> str:
    """Straight track heading due east at SPEED_MPS."""
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(ORIGIN_LAT))
    pts = []
    for i in range(int(DURATION_S) + 2):
        lon = ORIGIN_LON + (i * SPEED_MPS) / m_per_deg_lon
        stamp = (START_TIME + timedelta(seconds=i)).isoformat().replace("+00:00", "Z")
        pts.append(
            f'      <trkpt lat="{ORIGIN_LAT:.6f}" lon="{lon:.6f}">'
            f"<time>{stamp}</time></trkpt>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="PhysicalOS sample generator" '
        'xmlns="http://www.topografix.com/GPX/1/1">\n'
        "  <trk>\n    <name>PhysicalOS synthetic road sample</name>\n    <trkseg>\n"
        + "\n".join(pts)
        + "\n    </trkseg>\n  </trk>\n</gpx>\n"
    )


def build_route_yaml() -> str:
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(ORIGIN_LAT))
    end_lon = ORIGIN_LON + (DURATION_S * SPEED_MPS) / m_per_deg_lon
    return (
        "# Fallback route for videos with no GPS. Use with `pos ingest --route`.\n"
        "points:\n"
        f"  - [{ORIGIN_LAT:.6f}, {ORIGIN_LON:.6f}]\n"
        f"  - [{ORIGIN_LAT:.6f}, {end_lon:.6f}]\n"
    )


# ---------------------------------------------------------------------------


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = OUT_DIR / "frames_src"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir()

    n_render = int(DURATION_S * RENDER_FPS)
    print(f"Rendering {n_render} frames at {W}x{H} ...")
    for i in range(n_render):
        render_frame((i / RENDER_FPS) * SPEED_MPS).save(src / f"{i + 1:05d}.png")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_render}")

    video = OUT_DIR / "road.mp4"
    print("Encoding video ...")
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(RENDER_FPS),
            "-i", str(src / "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(video),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"ffmpeg failed:\n{proc.stderr}", file=sys.stderr)
        return 1

    shutil.rmtree(src)

    (OUT_DIR / "track.gpx").write_text(build_gpx())
    (OUT_DIR / "route.yaml").write_text(build_route_yaml())

    truth = build_truth()
    (OUT_DIR / "truth.json").write_text(json.dumps(truth, indent=1))

    # True world positions, so verify_sample.py can check the pipeline recovers
    # them. Heading is due east, so +lateral (right of travel) is SOUTH.
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(ORIGIN_LAT))
    (OUT_DIR / "objects.json").write_text(
        json.dumps(
            [
                {
                    "cls": o.cls,
                    "s_m": o.s,
                    "lateral_m": o.x,
                    "lat": ORIGIN_LAT - o.x / 111_320.0,
                    "lon": ORIGIN_LON + o.s / m_per_deg_lon,
                    "severity": o.severity,
                }
                for o in OBJECTS
            ],
            indent=2,
        )
    )

    n_items = sum(len(v) for v in truth.values())
    print(f"\nWrote {OUT_DIR}")
    print(f"  road.mp4      {DURATION_S:.0f}s, {RENDER_FPS} fps, {W}x{H}")
    print(f"  track.gpx     {int(DURATION_S) + 2} points, due east at {SPEED_MPS} m/s")
    print(f"  truth.json    {n_items} boxes across {len(truth)} keyframes")
    print(f"  objects.json  {len(OBJECTS)} world objects")
    print("\nNow run:")
    print("  uv run pos run --video samples/road/road.mp4 \\")
    print("      --gpx samples/road/track.gpx \\")
    print("      --truth samples/road/truth.json --out run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
