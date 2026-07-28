"""Geodesy and monocular ground-plane projection.

This is the module that turns "there is a pothole at pixel (412, 680)" into
"there is a pothole at 12.97164 N, 77.59472 E". Without it, findings are just
image annotations; with it, they are map assets.

Accuracy is roughly +/-2-4 m with a correctly measured camera height. That is
ample for placing markers convincingly on a map, and is honest about what
monocular geometry can do -- this is not survey-grade.
"""

from __future__ import annotations

import math

from .config import CameraConfig
from .schema import BOX_SCALE

# Metres per degree of latitude. Good to ~0.1% everywhere.
M_PER_DEG_LAT = 111_320.0


def m_per_deg_lon(lat: float) -> float:
    """Metres per degree of longitude shrinks with latitude."""
    return M_PER_DEG_LAT * math.cos(math.radians(lat))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to point 2, in [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def offset_latlon(
    lat: float, lon: float, north_m: float, east_m: float
) -> tuple[float, float]:
    """Move a lat/lon by a local ENU offset in metres."""
    return lat + north_m / M_PER_DEG_LAT, lon + east_m / m_per_deg_lon(lat)


def latlon_to_local_m(
    lat: float, lon: float, origin_lat: float, origin_lon: float
) -> tuple[float, float]:
    """Project to a local flat ENU frame in metres. Returns (east, north)."""
    return (
        (lon - origin_lon) * m_per_deg_lon(origin_lat),
        (lat - origin_lat) * M_PER_DEG_LAT,
    )


def focal_px(image_h: int, vfov_deg: float) -> float:
    """Pinhole focal length in pixels, from vertical field of view."""
    return (image_h / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)


def horizon_v(image_h: int, cam: CameraConfig) -> float:
    """Pixel row of the horizon. Anything at or above this has no ground range."""
    return image_h / 2.0 + cam.pitch_offset_frac * image_h


def project_to_ground(
    u: float,
    v: float,
    image_w: int,
    image_h: int,
    cam: CameraConfig,
) -> tuple[float, float] | None:
    """Project an image point onto the flat ground plane in front of the camera.

    Args:
        u, v: pixel coordinates, origin top-left.
        image_w, image_h: image size in pixels.
        cam: calibration.

    Returns:
        (forward_m, lateral_m) -- forward along the optical axis, lateral
        positive to the RIGHT. None when the point is at or above the horizon
        (it never meets the ground, so it has no ground range), or beyond
        cam.max_range_m where the projection is too ill-conditioned to trust.
    """
    f = focal_px(image_h, cam.vfov_deg)
    cx = image_w / 2.0
    cy = horizon_v(image_h, cam)

    dy = v - cy
    if dy <= 1e-6:
        return None  # at or above the horizon

    alpha = math.atan2(dy, f)  # angle below the optical axis
    forward = cam.height_m / math.tan(alpha)
    if forward <= 0 or forward > cam.max_range_m:
        return None

    lateral = (u - cx) * forward / f
    return forward, lateral


def ground_to_latlon(
    forward_m: float,
    lateral_m: float,
    cam_lat: float,
    cam_lon: float,
    heading_deg: float,
) -> tuple[float, float]:
    """Rotate a camera-relative ground offset into absolute lat/lon.

    heading_deg is the compass direction the camera faces (0 = north).
    """
    psi = math.radians(heading_deg)
    east = lateral_m * math.cos(psi) + forward_m * math.sin(psi)
    north = -lateral_m * math.sin(psi) + forward_m * math.cos(psi)
    return offset_latlon(cam_lat, cam_lon, north, east)


def box_ground_anchor(
    box: list[float], image_w: int, image_h: int, geometry: str = "point"
) -> tuple[float, float]:
    """The pixel where a detection meets the ground.

    `box` is [x1,y1,x2,y2] normalised to 0..BOX_SCALE, origin top-left.

    For a POINT defect the bottom edge is correct: a pothole's lower edge is
    where it touches the road, so that row gives its range.

    For an AREA or SEGMENT class the bottom edge is wrong. A waterlogged stretch
    or a length of edge cracking spans metres of road, and its box bottom is
    wherever the frame happens to cut it off -- often the frame's own bottom row,
    which projects to minimum range. Measured on real footage: 6 of 16 detections
    pinned at the 2.4 m floor exactly that way. The box CENTRE is a better
    estimate of where such a region actually lies.
    """
    x1, y1, x2, y2 = box
    u = ((x1 + x2) / 2.0) / BOX_SCALE * image_w
    v_norm = y2 if geometry == "point" else (y1 + y2) / 2.0
    return u, (v_norm / BOX_SCALE) * image_h


def bearing_to_box(
    box: list[float], image_w: int, image_h: int, cam: CameraConfig, heading_deg: float
) -> float:
    """Absolute compass bearing from the camera to a detection.

    This is the key quantity for triangulation, and the reason triangulation is
    worth doing at all: it depends ONLY on the box's horizontal centre. Not on
    the box bottom, not on camera pitch, not on the ground being flat -- all
    three of which `project_to_ground` depends on, and all three of which are its
    dominant error sources.

    A horizontal pixel offset from the image centre is an angle off the optical
    axis, atan((u - cx) / f). Add the camera heading and you have an absolute
    bearing, accurate to whatever the heading is accurate to.
    """
    x1, _y1, x2, _y2 = box
    u = ((x1 + x2) / 2.0) / BOX_SCALE * image_w
    f = focal_px(image_h, cam.vfov_deg)
    off = math.degrees(math.atan2(u - image_w / 2.0, f))
    return (heading_deg + off + 360.0) % 360.0


def triangulate(
    rays: list[tuple[float, float, float]],
) -> tuple[float, float, float, float] | None:
    """Least-squares intersection of bearing rays. The core geometry improvement.

    Each ray is (lat, lon, bearing_deg): a known camera position and the
    direction to the object. Two or more rays from DIFFERENT positions fix the
    object without any ground-plane assumption, which removes pitch error and
    ground flatness from the along-track error budget entirely.

    Returns (lat, lon, residual_m, parallax_deg), or None when the geometry
    cannot support a fix. The residual is a genuine geometric uncertainty rather
    than a model score, so it can be shown to a user as "+/-0.8 m" and meant
    literally.

    Solved in a local ENU tangent plane. A ray through point p with unit
    direction d gives the perpendicular-distance constraint

        n . x = n . p          where n is d rotated 90 degrees

    which is linear in x, so stacking rays is a plain 2x2 least-squares solve.
    Nearly parallel rays make it ill-conditioned, which is what `parallax`
    reports and why this returns None rather than a confident-looking number when
    every sighting came from effectively the same spot.
    """
    if len(rays) < 2:
        return None

    lat0 = sum(r[0] for r in rays) / len(rays)
    lon0 = sum(r[1] for r in rays) / len(rays)

    bearings = [r[2] % 360.0 for r in rays]
    parallax = max(
        abs((a - b + 180.0) % 360.0 - 180.0) for a in bearings for b in bearings
    )

    # Normal equations by hand: two unknowns, so a 2x2 solve is clearer and
    # faster than adding a linear-algebra dependency.
    saa = sab = sbb = sac = sbc = 0.0
    for lat, lon, brg in rays:
        e, n = latlon_to_local_m(lat, lon, lat0, lon0)
        psi = math.radians(brg)
        # Bearing 0 = +north, 90 = +east.
        de, dn = math.sin(psi), math.cos(psi)
        ne, nn = dn, -de  # normal to the ray, in (east, north)
        c = ne * e + nn * n
        saa += ne * ne
        sab += ne * nn
        sbb += nn * nn
        sac += ne * c
        sbc += nn * c

    det = saa * sbb - sab * sab
    if abs(det) < 1e-9:
        return None  # all rays parallel: nothing to intersect

    east = (sac * sbb - sbc * sab) / det
    north = (sbc * saa - sac * sab) / det

    # Residual: RMS perpendicular distance from the solution to each ray. This is
    # what says whether these sightings really saw one object.
    sq = 0.0
    for lat, lon, brg in rays:
        e, n = latlon_to_local_m(lat, lon, lat0, lon0)
        psi = math.radians(brg)
        ne, nn = math.cos(psi), -math.sin(psi)
        sq += (ne * (east - e) + nn * (north - n)) ** 2
    residual = math.sqrt(sq / len(rays))

    lat, lon = offset_latlon(lat0, lon0, north, east)
    return lat, lon, residual, parallax


def ray_forward_distance(
    lat: float, lon: float, brg: float, target_lat: float, target_lon: float
) -> float:
    """Signed distance along a ray to the point nearest `target`.

    Negative means the solution sits BEHIND the camera, which is impossible for
    something the camera photographed. A cheap sanity check on a triangulated
    fix: an intersection behind the lens means the association was wrong.
    """
    e, n = latlon_to_local_m(target_lat, target_lon, lat, lon)
    psi = math.radians(brg)
    return e * math.sin(psi) + n * math.cos(psi)


def polyline_length_m(points: list[tuple[float, float]]) -> float:
    return sum(
        haversine_m(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        for i in range(len(points) - 1)
    )
