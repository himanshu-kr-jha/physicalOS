"""Localise detections onto the map, then merge repeats into single findings.

This is the step that decides whether the output reads as an asset register or
as model spam. A pothole visible in six consecutive frames is ONE pothole with
six pieces of evidence -- not six potholes. Getting this wrong is the fastest
way to destroy trust in the whole system.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import CameraConfig, DomainConfig
from .geo import (
    bearing_to_box,
    box_ground_anchor,
    ground_to_latlon,
    haversine_m,
    project_to_ground,
    ray_forward_distance,
    triangulate,
)
from .perception.ensemble import iou
from .schema import Detection, Finding, Frame

# --------------------------------------------------------------------------
# Localise
# --------------------------------------------------------------------------


def localize_detections(
    detections: list[Detection],
    frames: list[Frame],
    cam: CameraConfig,
    domain: DomainConfig | None = None,
) -> list[Detection]:
    """Attach lat/lon/range to every detection that meets the ground plane.

    Detections above the horizon cannot be ranged by ground-plane projection
    (a streetlight head, a facade crack three storeys up). Those are still
    real observations, so we pin them at the camera position and leave
    `range_m` unset -- downstream code can tell they were never truly ranged.

    `domain` selects the anchor row per class. Point defects project from the box
    BOTTOM, where they meet the road. Area and segment classes project from the
    box CENTRE, because their bottom edge is usually just where the frame cut
    them off, which lands them at minimum range. Omitting the domain keeps the
    old bottom-edge behaviour for every class.
    """
    by_id = {f.frame_id: f for f in frames}

    for det in detections:
        frame = by_id.get(det.frame_id)
        if frame is None:
            continue

        geom = domain.spec(det.cls).geometry if domain else "point"
        det.anchor = "bottom" if geom == "point" else "centre"
        u, v = box_ground_anchor(det.box, frame.width, frame.height, geom)
        ground = project_to_ground(u, v, frame.width, frame.height, cam)

        if ground is None:
            det.lat, det.lon, det.range_m = frame.lat, frame.lon, None
            continue

        forward, lateral = ground
        det.lat, det.lon = ground_to_latlon(
            forward, lateral, frame.lat, frame.lon, frame.heading_deg
        )
        det.range_m = round(forward, 2)

    return detections


# --------------------------------------------------------------------------
# Cluster
# --------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# --------------------------------------------------------------------------
# Temporal association
#
# WHY TRACK BEFORE CLUSTERING
# Spatial clustering alone fails exactly when it matters most. Measured on real
# Kohima footage, two independent projections of ONE object disagree by a median
# 2.46 m (p90 6.26 m). The pothole cluster radius is 3.0 m. So two sightings of
# one pothole routinely land further apart than the radius allows and become TWO
# findings -- 13 of 30 multi-sighting findings were within a whisker of splitting.
# Raising the radius is not a fix: it would start merging genuinely separate
# potholes.
#
# The way out is that consecutive keyframes are a couple of metres apart, so the
# same object's box moves PREDICTABLY between them -- downward and outward as it
# approaches. Matching on that is independent of the position error, so it
# associates correctly even when the projected positions disagree.
# --------------------------------------------------------------------------

# Minimum box overlap to consider two detections the same object. Deliberately
# low: an object approaching at 2 m per keyframe grows and shifts a lot, so
# demanding tracker-grade IoU would break every real track.
MIN_IOU = 0.10

# How many keyframes a track may skip. 1 lets a track survive a single missed
# detection -- common at 48% recall -- without bridging unrelated objects.
MAX_GAP = 1


def _centre(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _assoc_score(prev: Detection, cur: Detection) -> float:
    """How strongly do these two detections look like one object, one frame apart?

    Combines overlap with the physical prior that an approaching object moves
    DOWN the image. Returns -1 when the pair should not be linked.
    """
    ov = iou(prev.box, cur.box)
    (pcx, pcy), (ccx, ccy) = _centre(prev.box), _centre(cur.box)

    # Things the camera approaches move down the frame. A candidate that moved
    # UP appreciably is a different object, so refuse it outright.
    if ccy < pcy - 60.0:
        return -1.0

    # Horizontal drift, normalised by box width: an object should not jump
    # across the frame between adjacent keyframes.
    w = max(prev.box[2] - prev.box[0], 40.0)
    dx = abs(ccx - pcx) / w
    if dx > 2.5:
        return -1.0

    if ov < MIN_IOU and dx > 1.0:
        return -1.0

    return ov + max(0.0, 1.0 - dx / 2.5)


def build_tracks(
    detections: list[Detection], frames: list[Frame]
) -> list[list[Detection]]:
    """Group detections of one class into per-object tracks across keyframes."""
    order = {f.frame_id: i for i, f in enumerate(frames)}
    by_class: dict[str, list[Detection]] = {}
    for d in detections:
        if d.frame_id in order:
            by_class.setdefault(d.cls, []).append(d)

    tracks: list[list[Detection]] = []
    for dets in by_class.values():
        dets.sort(key=lambda d: order[d.frame_id])
        open_tracks: list[list[Detection]] = []

        for det in dets:
            idx = order[det.frame_id]
            best, best_score = -1, 0.0
            for i, tr in enumerate(open_tracks):
                last = tr[-1]
                gap = idx - order[last.frame_id]
                if gap < 1 or gap > MAX_GAP + 1:
                    continue  # same frame, or too long a silence
                s = _assoc_score(last, det)
                if s > best_score:
                    best, best_score = i, s
            if best >= 0:
                open_tracks[best].append(det)
            else:
                open_tracks.append([det])

        tracks.extend(open_tracks)
    return tracks


def _fix_from_rays(
    track: list[Detection], frames_by_id: dict[str, Frame], cam: CameraConfig
) -> tuple[float, float, float, float, int] | None:
    """Triangulate a track. Returns (lat, lon, residual, parallax, n_rays)."""
    rays: list[tuple[float, float, float]] = []
    for det in track:
        fr = frames_by_id.get(det.frame_id)
        if fr is None:
            continue
        rays.append(
            (
                fr.lat,
                fr.lon,
                bearing_to_box(det.box, fr.width, fr.height, cam, fr.heading_deg),
            )
        )
    if len(rays) < 2:
        return None

    got = triangulate(rays)
    if got is None:
        return None
    lat, lon, residual, parallax = got

    # Reject a fix behind the camera: impossible for something photographed, so
    # the association must be wrong.
    for r_lat, r_lon, brg in rays:
        if ray_forward_distance(r_lat, r_lon, brg, lat, lon) < 0.0:
            return None

    return lat, lon, residual, parallax, len(rays)


def cluster_detections(
    detections: list[Detection],
    frames: list[Frame],
    domain: DomainConfig,
    cam: CameraConfig | None = None,
    min_parallax_deg: float = 8.0,
    max_residual_m: float = 4.0,
) -> list[Finding]:
    """Merge repeated sightings of one object into one Finding.

    Two stages. First TRACK: associate detections across adjacent keyframes by
    box overlap and motion, which is immune to the position error that defeats
    pure spatial clustering. Then CLUSTER whatever remains unassociated, using
    the original greedy per-class radius, so a genuinely re-sighted object that
    the tracker missed still merges.

    When `cam` is supplied, a track of two or more sightings is positioned by
    TRIANGULATION rather than by averaging its projections -- see pos/geo.py.
    The ground plane stays as the fallback whenever the geometry is too weak
    (`min_parallax_deg`) or the rays disagree (`max_residual_m`), so a worse
    number never silently replaces a better one.
    """
    t_by_frame = {f.frame_id: f.t_sec for f in frames}
    frames_by_id = {f.frame_id: f for f in frames}

    # --- stage 1: temporal tracks -----------------------------------------
    # Each track is one object as followed across adjacent keyframes.
    clusters: list[list[Detection]] = build_tracks(detections, frames)

    # --- stage 2: spatial merge over ALL clusters --------------------------
    # Not just singletons into tracks. A track can be broken in two by a missed
    # detection longer than MAX_GAP, and those two halves are the same object --
    # if only singletons were merged here, they would stay two findings and the
    # duplicate count would be WORSE than pure spatial clustering. Measured that
    # regression: kohima4 duplicates went 2 -> 6 before this pass existed.
    def centroid_of(cluster: list[Detection]) -> tuple[float, float] | None:
        lats = [d.lat for d in cluster if d.lat is not None]
        lons = [d.lon for d in cluster if d.lon is not None]
        return (_median(lats), _median(lons)) if lats else None

    clusters.sort(key=lambda c: t_by_frame.get(c[0].frame_id, 0.0))
    merged = True
    while merged:
        merged = False
        cents = [centroid_of(c) for c in clusters]
        for i in range(len(clusters)):
            if cents[i] is None:
                continue
            radius = domain.spec(clusters[i][0].cls).cluster_radius_m
            for j in range(i + 1, len(clusters)):
                if cents[j] is None or clusters[j][0].cls != clusters[i][0].cls:
                    continue
                if haversine_m(*cents[i], *cents[j]) <= radius:  # type: ignore[misc]
                    clusters[i] = clusters[i] + clusters[j]
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break

    # --- build findings ----------------------------------------------------
    findings: list[Finding] = []
    ranked = sorted(clusters, key=lambda c: t_by_frame.get(c[0].frame_id, 0.0))

    for i, cluster in enumerate(ranked):
        spec = domain.spec(cluster[0].cls)
        lats = [d.lat for d in cluster if d.lat is not None]
        lons = [d.lon for d in cluster if d.lon is not None]

        lat = _median(lats) if lats else None
        lon = _median(lons) if lons else None
        method, residual, parallax, n_rays = "ground_plane", None, None, 0
        if lat is None:
            method = "camera"

        if cam is not None and len(cluster) >= 2:
            fix = _fix_from_rays(cluster, frames_by_id, cam)
            if fix is not None:
                t_lat, t_lon, res, par, n = fix
                n_rays, parallax, residual = n, round(par, 2), round(res, 2)
                if par >= min_parallax_deg and res <= max_residual_m:
                    lat, lon, method = t_lat, t_lon, "triangulated"

        mean_conf = sum(d.confidence for d in cluster) / len(cluster)
        agreement = min(1.0, 0.75 + 0.08 * (len(cluster) - 1))
        confidence = mean_conf * agreement

        # A triangulated fix whose rays nearly meet is genuine corroboration
        # from two viewpoints; reflect that in the confidence rather than only
        # in a separate field nobody reads.
        if method == "triangulated" and residual is not None and residual < 1.0:
            confidence = min(0.99, confidence * 1.10)

        findings.append(
            Finding(
                finding_id=f"f-{i:04d}",
                cls=cluster[0].cls,
                label=spec.label,
                geometry=spec.geometry,  # type: ignore[arg-type]
                lat=lat,
                lon=lon,
                severity=int(round(_median([float(d.severity) for d in cluster]))),
                confidence=round(max(0.0, min(1.0, confidence)), 3),
                t_sec=min(t_by_frame.get(d.frame_id, 0.0) for d in cluster),
                evidence=cluster,
                pos_method=method,
                pos_residual_m=residual,
                n_rays=n_rays,
                parallax_deg=parallax,
            )
        )
    return findings


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def write_detections(detections: list[Detection], path: Path) -> None:
    """NDJSON so a long run can be appended to and survives a crash."""
    with open(path, "w") as fh:
        for d in detections:
            fh.write(json.dumps(d.model_dump()) + "\n")


def read_detections(path: Path) -> list[Detection]:
    out: list[Detection] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Detection(**json.loads(line)))
    return out


def write_findings(findings: list[Finding], path: Path) -> None:
    Path(path).write_text(json.dumps([f.model_dump() for f in findings], indent=2))


def read_findings(path: Path) -> list[Finding]:
    return [Finding(**d) for d in json.loads(Path(path).read_text())]
