"""Fetch OpenStreetMap context geometry for the route: buildings and roads.

map3d gets this from `api.fleet.cartesiancs.com`, which is Cartesian CS's own
backend and not usable here, so we query Overpass directly.

Overpass is slow and rate limited, so every response is cached on disk. A
missing or refused response is never fatal -- the viewer degrades to the route
and findings alone, which still demos.

OSM heights are frequently missing or wrong (map3d's own README says as much).
We fall back to levels x 3.2 m, then to a modest default.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .schema import Building, Frame, RoadWay, Twin

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

METRES_PER_LEVEL = 3.2
DEFAULT_HEIGHT_M = 8.0

# Only ways worth drawing. Service roads and tracks would bloat the payload.
ROAD_TYPES = (
    "motorway|trunk|primary|secondary|tertiary|residential|unclassified|"
    "living_street|pedestrian|footway"
)


def route_bbox(
    frames: list[Frame], margin_m: float = 120.0
) -> tuple[float, float, float, float]:
    """(south, west, north, east) around the route, padded by margin_m."""
    lats = [f.lat for f in frames]
    lons = [f.lon for f in frames]

    dlat = margin_m / 111_320.0
    mid = sum(lats) / len(lats)
    dlon = margin_m / (111_320.0 * max(math.cos(math.radians(mid)), 1e-6))

    return (min(lats) - dlat, min(lons) - dlon, max(lats) + dlat, max(lons) + dlon)


def _query(bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    b = f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"
    return f"""
[out:json][timeout:60];
(
  way["building"]({b});
  way["highway"~"^({ROAD_TYPES})$"]({b});
);
out body geom;
""".strip()


def _fetch(bbox: tuple[float, float, float, float], cache_dir: Path | None) -> dict:
    body = _query(bbox)

    cache: Path | None = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(body.encode()).hexdigest()[:16]
        cache = cache_dir / f"overpass_{digest}.json"
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except json.JSONDecodeError:
                pass

    import httpx

    last: Exception | None = None
    for url in OVERPASS_URLS:
        try:
            resp = httpx.post(url, data={"data": body}, timeout=90.0)
            resp.raise_for_status()
            data = resp.json()
            if cache:
                cache.write_text(json.dumps(data))
            return data
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last = exc
    raise RuntimeError(f"All Overpass mirrors failed: {last}")


def _height_of(tags: dict) -> float:
    """Best-effort building height. OSM data here is famously patchy."""
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        try:
            # "12", "12 m", "12.5m" all occur in the wild.
            return max(2.0, float(str(raw).lower().replace("m", "").strip()))
        except ValueError:
            pass

    levels = tags.get("building:levels") or tags.get("levels")
    if levels:
        try:
            return max(2.0, float(str(levels).split(";")[0].strip()) * METRES_PER_LEVEL)
        except ValueError:
            pass

    return DEFAULT_HEIGHT_M


_KEEP_TAGS = (
    "building",
    "name",
    "height",
    "building:levels",
    "amenity",
    "addr:street",
    "addr:housenumber",
)


def parse_overpass(data: dict) -> Twin:
    buildings: list[Building] = []
    roads: list[RoadWay] = []

    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        pts = [[g["lat"], g["lon"]] for g in geom if "lat" in g and "lon" in g]
        if len(pts) < 2:
            continue
        tags = el.get("tags") or {}

        if "building" in tags:
            if len(pts) < 4:
                continue  # not a closed footprint worth extruding
            buildings.append(
                Building(
                    osm_id=int(el["id"]),
                    height_m=round(_height_of(tags), 2),
                    footprint=pts,
                    tags={k: str(v) for k, v in tags.items() if k in _KEEP_TAGS},
                )
            )
        elif "highway" in tags:
            roads.append(
                RoadWay(
                    osm_id=int(el["id"]),
                    highway=str(tags["highway"]),
                    name=str(tags.get("name", "")),
                    path=pts,
                )
            )

    return Twin(buildings=buildings, roads=roads)


def build_twin(
    frames: list[Frame],
    out_path: Path,
    margin_m: float = 120.0,
    offline: bool = False,
) -> Twin:
    """Fetch and write context geometry. Returns an empty Twin on failure."""
    out_path = Path(out_path)

    if offline:
        twin = Twin()
    else:
        bbox = route_bbox(frames, margin_m=margin_m)
        try:
            twin = parse_overpass(_fetch(bbox, out_path.parent / ".cache"))
        except Exception:  # noqa: BLE001 - never fatal; the viewer copes
            twin = Twin()

    out_path.write_text(json.dumps(twin.model_dump(), indent=2))
    return twin
