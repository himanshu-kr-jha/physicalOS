"""Fetch and stitch satellite imagery under the route, once, server-side.

WHY
The 3D scene draws the route on a dark grey plane with grey OSM boxes. That is
legible but abstract, and the abstraction hides the one thing a viewer most wants
to judge: is that pothole on the road, or in a field beside it? Real imagery
underneath answers it instantly.

WHY SERVER-SIDE AND STITCHED
The obvious approach -- a tile pyramid in three.js -- needs a tile loader, LOD
juggling and hundreds of requests per page view. A route is a few hundred metres,
so the whole thing fits in ONE texture. Fetch once, stitch once, cache in the run
directory like every other artefact, and the viewer loads a single JPEG. It also
puts the licence-sensitive fetch in one auditable place rather than in every
browser that opens the dashboard.

LICENSING -- READ THIS
Tile imagery is not public domain and providers differ sharply:

  esri    Esri World Imagery. Free for NON-COMMERCIAL use with attribution;
          commercial use needs an ArcGIS licence. Default here because this
          project is stated to be academic and because it has worldwide
          sub-metre coverage, rural India included.
  osm     OpenStreetMap standard tiles. ODbL, attribution required. A STREET map,
          not satellite -- useful for checking alignment, not for looking real.
          Heavy automated use breaches the tile usage policy.
  mapbox  Mapbox Satellite. Needs an access token; clear commercial path.

Google's tiles are deliberately absent: they may not be used outside Google's own
APIs, so there is no lawful way to put them in this texture. Use `pos kml` and
Google Earth for Google imagery instead.

Whatever the provider, `attribution` travels in basemap.json and the viewer
renders it. Dropping it is a licence breach, not a cosmetic choice.

ACCURACY
Tiles are Web Mercator, whose y axis is nonlinear in latitude, but a run spans a
few hundred metres and the resulting error is far below one pixel. The imagery's
own georeferencing is the real limit -- it can sit a metre or two off -- so a
small constant offset between route and road is the provider's, not this
pipeline's.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from .geo import latlon_to_local_m
from .schema import Frame

TILE_PX = 256

PROVIDERS: dict[str, dict[str, str]] = {
    "esri": {
        "url": (
            "https://services.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attribution": "Imagery © Esri, Maxar, Earthstar Geographics",
        "kind": "satellite",
        "licence": "Non-commercial use with attribution; commercial needs an ArcGIS licence.",
    },
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors (ODbL)",
        "kind": "street",
        "licence": "ODbL. Respect the OSM tile usage policy -- do not bulk-fetch.",
    },
    "mapbox": {
        "url": (
            "https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg90"
            "?access_token={token}"
        ),
        "attribution": "© Mapbox © Maxar",
        "kind": "satellite",
        "licence": "Requires an access token. Free tier available.",
    },
}

# A run spans a few hundred metres; more than this means a wrong zoom or a
# corrupt track, and hammering a tile server is both rude and rate-limited.
MAX_TILES = 256


class BasemapError(RuntimeError):
    pass


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Web Mercator tile coordinates, fractional."""
    n = 2.0**z
    x = (lon + 180.0) / 360.0 * n
    rad = math.radians(max(-85.05112878, min(85.05112878, lat)))
    y = (1.0 - math.asinh(math.tan(rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lon(x: float, z: int) -> float:
    return x / 2.0**z * 360.0 - 180.0


def tile_to_lat(y: float, z: int) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / 2.0**z))))


def _pad_bbox(
    frames: list[Frame], margin_m: float
) -> tuple[float, float, float, float]:
    """Route bounding box in degrees, padded by `margin_m` on every side."""
    lats = [f.lat for f in frames]
    lons = [f.lon for f in frames]
    lat0 = sum(lats) / len(lats)
    dlat = margin_m / 111_320.0
    dlon = margin_m / (111_320.0 * max(math.cos(math.radians(lat0)), 1e-6))
    return min(lats) - dlat, max(lats) + dlat, min(lons) - dlon, max(lons) + dlon


def build_basemap(
    frames: list[Frame],
    origin: tuple[float, float],
    out_dir: Path,
    provider: str = "esri",
    zoom: int = 18,
    margin_m: float = 60.0,
    url_template: str | None = None,
    attribution: str | None = None,
    token: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Fetch tiles covering the route, stitch to one JPEG, write the bounds JSON."""
    if len(frames) < 2:
        raise BasemapError("Need at least two keyframes to compute a route bbox.")

    import httpx
    from PIL import Image

    preset = PROVIDERS.get(provider, {})
    url = url_template or os.environ.get("POS_TILE_URL") or preset.get("url")
    if not url:
        raise BasemapError(
            f"Unknown provider {provider!r} and no --tiles/POS_TILE_URL given. "
            f"Known: {', '.join(PROVIDERS)}"
        )
    credit = (
        attribution
        or os.environ.get("POS_TILE_ATTRIBUTION")
        or preset.get("attribution")
        or provider
    )
    token = token or os.environ.get("POS_TILE_TOKEN") or ""
    if "{token}" in url and not token:
        raise BasemapError(
            f"Provider {provider!r} needs a token: pass --token or set POS_TILE_TOKEN."
        )

    lat_min, lat_max, lon_min, lon_max = _pad_bbox(frames, margin_m)

    x0f, y0f = lonlat_to_tile(lon_min, lat_max, zoom)  # NW corner
    x1f, y1f = lonlat_to_tile(lon_max, lat_min, zoom)  # SE corner
    x0, y0 = int(math.floor(x0f)), int(math.floor(y0f))
    x1, y1 = int(math.floor(x1f)), int(math.floor(y1f))

    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    if nx * ny > MAX_TILES:
        raise BasemapError(
            f"{nx}x{ny} = {nx * ny} tiles exceeds the {MAX_TILES} cap. "
            f"Lower --zoom -- each step down quarters the count."
        )

    canvas = Image.new("RGB", (nx * TILE_PX, ny * TILE_PX), (32, 38, 46))
    fetched = failed = 0
    headers = {"User-Agent": "PhysicalOS/0.1 (road inspection POC)"}

    with httpx.Client(
        timeout=timeout, follow_redirects=True, headers=headers
    ) as client:
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                tile_url = (
                    url.replace("{z}", str(zoom))
                    .replace("{x}", str(tx))
                    .replace("{y}", str(ty))
                    .replace("{token}", token)
                )
                try:
                    r = client.get(tile_url)
                    r.raise_for_status()
                    with Image.open(BytesIO(r.content)) as im:
                        # Some providers return 512 px tiles (@2x); resize so the
                        # canvas grid stays consistent.
                        tile = im.convert("RGB")
                        if tile.size != (TILE_PX, TILE_PX):
                            tile = tile.resize((TILE_PX, TILE_PX), Image.LANCZOS)
                        canvas.paste(
                            tile, ((tx - x0) * TILE_PX, (ty - y0) * TILE_PX)
                        )
                    fetched += 1
                except Exception:  # noqa: BLE001 - one bad tile must not lose the rest
                    failed += 1

    if fetched == 0:
        raise BasemapError(
            f"No tiles fetched from {provider!r}. Check the URL template and network."
        )

    # Exact geographic extent of the STITCHED image -- whole tiles, not the
    # requested bbox. Using the request bbox would misalign the texture by up to
    # one tile, which reads as bad localisation.
    img_lon_min = tile_to_lon(x0, zoom)
    img_lon_max = tile_to_lon(x1 + 1, zoom)
    img_lat_max = tile_to_lat(y0, zoom)
    img_lat_min = tile_to_lat(y1 + 1, zoom)

    # Local ENU metres, matching toLocal() in viewer/src/store.ts: x = east,
    # z = -north. So the image's NORTH edge is the MOST NEGATIVE z.
    e_min, n_min = latlon_to_local_m(img_lat_min, img_lon_min, origin[0], origin[1])
    e_max, n_max = latlon_to_local_m(img_lat_max, img_lon_max, origin[0], origin[1])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / "basemap.jpg", "JPEG", quality=88, optimize=True)

    meta = {
        "provider": provider,
        "attribution": credit,
        "kind": preset.get("kind", "unknown"),
        "zoom": zoom,
        "size": {"w": canvas.width, "h": canvas.height},
        "tiles": {
            "x0": x0,
            "y0": y0,
            "nx": nx,
            "ny": ny,
            "fetched": fetched,
            "failed": failed,
        },
        "bbox": {
            "lat_min": img_lat_min,
            "lat_max": img_lat_max,
            "lon_min": img_lon_min,
            "lon_max": img_lon_max,
        },
        # What the viewer actually needs: where to put the plane, in metres.
        "local": {
            "x_min": e_min,
            "x_max": e_max,
            "z_min": -n_max,
            "z_max": -n_min,
            "width_m": e_max - e_min,
            "height_m": n_max - n_min,
        },
        "m_per_px": (e_max - e_min) / max(canvas.width, 1),
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "basemap.json").write_text(json.dumps(meta, indent=2))
    return meta
