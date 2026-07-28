"""Export a run as a self-contained KMZ for Google Earth.

WHY THIS EXISTS
The dashboard is the analysis tool, but it draws its own world: OSM boxes on a
grey plane. Google Earth already has the real one -- worldwide satellite imagery
and 3D terrain, free -- so pointing our findings at it immediately answers "is
that pothole really on that road?" in a way a synthetic scene cannot.

Google Earth Pro also has Movie Maker (Tools > Movie Maker), which records a
scripted flythrough straight to MP4. That is a better demo asset than screen-
capturing a browser, which is why this writer emits a <gx:Tour>.

WHAT IT DELIBERATELY DOES NOT DO
Google's Photorealistic 3D Tiles cover ~2,500 cities, high-density urban only, so
rural footage gains nothing from them. This export relies only on what exists
everywhere: imagery plus terrain. The point cloud is not exported either -- KML
has no representation for 400,000 loose points. The cloud stays in the dashboard.

TWO CONVENTIONS THAT BITE
  1. KML colours are aabbggrr, NOT #rrggbb. Getting this wrong silently swaps red
     and blue, so a green->red heatmap reads inverted -- greens exactly where the
     worst potholes are. _kml_color is the only place that conversion happens.
  2. KML coordinates are lon,lat,alt -- longitude FIRST, the opposite of every
     other file in this project. _coord is the only place that order is written.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from .absence import absence_gaps, gap_polyline
from .config import DomainConfig
from .schema import BOX_SCALE, Detection, Finding, Frame, Segment

# Popup images are downscaled to this width. Full frames are ~300 KB each and a
# balloon is a few hundred pixels wide, so shipping originals would multiply the
# archive size for pixels nobody sees.
POPUP_WIDTH = 640
ICON_PX = 64

# Literal U+00B7, not "&middot;". XML predefines only &lt; &gt; &amp; &quot;
# &apos; -- every other HTML entity is undefined and makes the document
# unparseable. Google Earth tolerates it; QGIS and ArcGIS do not. Entities are
# safe inside the CDATA balloon HTML, but never in element text like <name>.
SEP = "·"


def _kml_color(hex_color: str, alpha: int = 255) -> str:
    """#rrggbb -> aabbggrr, KML's byte order.

    KML stores colour as ABGR. Passing #rrggbb straight through swaps red and
    blue, which on a green->red quality ramp inverts the entire reading.
    """
    h = (hex_color or "").lstrip("#")
    if len(h) == 3:  # #abc shorthand
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        h = "f59e0b"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"{alpha:02x}{b}{g}{r}"


def _coord(lat: float, lon: float, alt: float = 0.0) -> str:
    """KML coordinate triple. Longitude FIRST -- the opposite of our own files."""
    return f"{lon:.7f},{lat:.7f},{alt:g}"


def _esc(text: object) -> str:
    return escape(str(text if text is not None else ""))


def _cdata(html: str) -> str:
    # "]]>" inside the payload would close the section early.
    return f"<![CDATA[{html.replace(']]>', ']]&gt;')}]]>"


def _kml_time(ts: str | None) -> str | None:
    """Trim an ISO-8601 stamp to whole seconds.

    Frame.ts already has the right format and, importantly, the right clock --
    GPS wall time, not video time. Only the microseconds some readers dislike
    are dropped.
    """
    if not ts:
        return None
    if "." not in ts:
        return ts
    head, _, tail = ts.partition(".")
    if tail.endswith("Z"):
        return head + "Z"
    for sign in ("+", "-"):
        if sign in tail:
            return head + tail[tail.index(sign) :]
    return head + "Z"


def _best_sighting(evidence: list[Detection]) -> Detection | None:
    """The largest box: the closest, most legible pass.

    Evidence is in time order, so evidence[0] is the FIRST sighting -- furthest
    away, often a few-pixel sliver. Mirrors bestSighting() in
    viewer/src/EvidencePanel.tsx, so the KMZ and the dashboard agree on which
    frame represents a finding.
    """
    if not evidence:
        return None
    return max(evidence, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))


def _pin_png(hollow: bool = False) -> bytes:
    """A WHITE disc (or ring), tinted per class by <IconStyle><color>.

    White because IconStyle colour MULTIPLIES the texture: a white source takes
    any tint exactly, a coloured source would muddy it. Generated rather than
    linked because maps.google.com/mapfiles icons need network access, and a KMZ
    that only renders online defeats the point of packing one.
    """
    from PIL import Image, ImageDraw

    s = ICON_PX
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 4
    if hollow:
        # Absence is a property of a stretch, not a spot, so a ring reads as
        # "along here" rather than "exactly here".
        d.ellipse([pad, pad, s - pad, s - pad], outline=(255, 255, 255, 255), width=7)
    else:
        d.ellipse([pad, pad, s - pad, s - pad], fill=(255, 255, 255, 255))
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _evidence_jpeg(frame_path: Path, det: Detection, draw_box: bool) -> bytes | None:
    """The evidence frame with its box burned in, downscaled for a balloon.

    Burned in rather than overlaid with HTML because a KML balloon cannot
    position an element over an image reliably across Google Earth Pro, Earth Web
    and QGIS. Box coordinates use BOX_SCALE from pos/schema.py -- the convention
    is not re-derived here.
    """
    if not frame_path.exists():
        return None
    from PIL import Image, ImageDraw

    try:
        with Image.open(frame_path) as im:
            img = im.convert("RGB")
            w, h = img.size

            if draw_box:
                x1 = det.box[0] / BOX_SCALE * w
                y1 = det.box[1] / BOX_SCALE * h
                x2 = det.box[2] / BOX_SCALE * w
                y2 = det.box[3] / BOX_SCALE * h
                d = ImageDraw.Draw(img)
                # Dark outer stroke first: a single bright rectangle disappears
                # against pale tarmac.
                d.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=6)
                d.rectangle([x1, y1, x2, y2], outline=(255, 210, 60), width=3)

            if w > POPUP_WIDTH:
                img = img.resize(
                    (POPUP_WIDTH, max(1, round(h * POPUP_WIDTH / w))), Image.LANCZOS
                )
            buf = BytesIO()
            img.save(buf, "JPEG", quality=82, optimize=True)
            return buf.getvalue()
    except OSError:
        return None


def _balloon(
    f: Finding,
    det: Detection | None,
    img_name: str | None,
    absent: bool,
    index_name: str,
    approx_pos: bool = False,
) -> str:
    """Popup HTML for one finding."""
    rows: list[tuple[str, str]] = [
        ("Severity", f"{f.severity} of 5"),
        ("Confidence", f"{f.confidence * 100:.0f}%"),
        ("Sightings", str(len(f.evidence))),
    ]
    if f.lat is not None and f.lon is not None:
        rows.append(("Position", f"{f.lat:.6f}, {f.lon:.6f}"))
    elif approx_pos:
        rows.append(("Position", "camera position — not ranged"))
    if det is not None:
        rows.append(
            (
                "Range at sighting",
                f"{det.range_m:.1f} m ahead"
                if det.range_m is not None
                else "above horizon — not ranged",
            )
        )
        rows.append(("Frame", _esc(det.frame_id)))
    rows.append(("ID", _esc(f.finding_id)))

    table = "".join(
        f'<tr><td style="padding:2px 10px 2px 0;color:#666;white-space:nowrap">{k}</td>'
        f'<td style="padding:2px 0"><b>{v}</b></td></tr>'
        for k, v in rows
    )

    parts = [f'<div style="font-family:Arial,sans-serif;max-width:{POPUP_WIDTH}px">']

    if absent:
        parts.append(
            '<div style="background:#3f3f46;color:#fff;padding:8px 10px;'
            'margin-bottom:8px;border-radius:3px">'
            "<b>NOT DETECTED ANYWHERE ALONG THIS STRETCH</b><br/>"
            "Inferred from coverage, not seen in a frame. The image below is "
            "simply the middle of the gap, so there is no box to draw."
            "</div>"
        )

    if img_name:
        parts.append(
            f'<img src="{img_name}" width="{POPUP_WIDTH}" '
            'style="max-width:100%;border-radius:3px"/>'
        )

    if det is not None and det.evidence:
        parts.append(
            '<blockquote style="margin:10px 0;padding:8px 12px;'
            "border-left:3px solid #999;background:#f4f4f5;font-style:italic\">"
            f"{_esc(det.evidence)}</blockquote>"
        )

    parts.append(f'<table style="font-size:13px;margin-top:6px">{table}</table>')
    parts.append(
        '<div style="margin-top:10px;font-size:11px;color:#888">'
        f"PhysicalOS &middot; {_esc(index_name)} &middot; position from monocular "
        "ground-plane projection, &plusmn;2&ndash;4 m. Not survey grade.</div>"
    )
    parts.append("</div>")
    return "".join(parts)


def _look_at(lat: float, lon: float, range_m: float = 120.0, tilt: float = 62.0) -> str:
    return (
        "<LookAt>"
        f"<longitude>{lon:.7f}</longitude><latitude>{lat:.7f}</latitude>"
        f"<altitude>0</altitude><heading>0</heading><tilt>{tilt:g}</tilt>"
        f"<range>{range_m:g}</range>"
        "<altitudeMode>relativeToGround</altitudeMode>"
        "</LookAt>"
    )


def build_kmz(
    run_dir: Path,
    out_path: Path,
    tour: bool = True,
    max_tour_stops: int = 40,
) -> dict:
    """Write a self-contained KMZ. Returns a summary dict."""
    run_dir = Path(run_dir)
    out_path = Path(out_path)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    frames = [Frame(**f) for f in json.loads((run_dir / "frames.json").read_text())]
    findings = [
        Finding(**f) for f in json.loads((run_dir / "findings.json").read_text())
    ]

    seg_path = run_dir / "segments.json"
    segments = (
        [Segment(**s) for s in json.loads(seg_path.read_text())]
        if seg_path.exists()
        else []
    )

    try:
        domain = DomainConfig.load(manifest.get("domain", "road"))
    except FileNotFoundError:
        domain = DomainConfig(key="road", label="Road")
    class_map = domain.class_map
    absence_keys = {a.key for a in domain.absences}
    index_name = domain.index_name

    # Findings carry t_sec (video clock); the time slider needs wall-clock, which
    # lives on Frame.ts (GPS clock). Map through the nearest keyframe.
    def wall_time(t_sec: float) -> str | None:
        if not frames:
            return None
        return _kml_time(min(frames, key=lambda fr: abs(fr.t_sec - t_sec)).ts)

    # Absence findings store only their midpoint but describe a stretch. Recover
    # the extent from the same helper that produced them.
    present_absence = {f.cls for f in findings} & absence_keys
    presence_only = [f for f in findings if f.cls not in absence_keys]
    gaps = (
        absence_gaps(frames, presence_only, domain, detectable=None)
        if present_absence
        else []
    )

    def gap_for(f: Finding):
        """The gap whose midpoint is nearest this absence finding."""
        best, best_d = None, float("inf")
        for g in gaps:
            if g.rule.key != f.cls or f.lat is None or f.lon is None:
                continue
            d = (g.mid_lat - f.lat) ** 2 + (g.mid_lon - f.lon) ** 2
            if d < best_d:
                best, best_d = g, d
        return best

    # ---- styles -----------------------------------------------------------
    styles: list[str] = []
    for key, spec in class_map.items():
        absent = key in absence_keys
        styles.append(
            f'<Style id="c-{_esc(key)}">'
            "<IconStyle>"
            f"<color>{_kml_color(spec.color)}</color>"
            f"<scale>{1.4 if absent else 1.1:g}</scale>"
            f"<Icon><href>{'img/gap.png' if absent else 'img/pin.png'}</href></Icon>"
            "</IconStyle>"
            "<LabelStyle><scale>0</scale></LabelStyle>"
            f"<LineStyle><color>{_kml_color(spec.color, 200)}</color>"
            "<width>7</width></LineStyle>"
            "<BalloonStyle><text>$[description]</text></BalloonStyle>"
            "</Style>"
        )
    styles.append(
        '<Style id="route">'
        "<LineStyle><color>ffe6edf3</color><width>3</width></LineStyle>"
        "<IconStyle><scale>0.7</scale><Icon><href>img/pin.png</href></Icon></IconStyle>"
        "<LabelStyle><scale>0</scale></LabelStyle>"
        "</Style>"
    )

    # ---- findings ---------------------------------------------------------
    images: dict[str, bytes] = {}
    placemarks: list[str] = []
    gap_lines: list[str] = []
    skipped = 0

    for f in sorted(findings, key=lambda x: x.t_sec):
        absent = f.cls in absence_keys
        det = _best_sighting(f.evidence)
        spec = class_map.get(f.cls)
        label = f.label or (spec.label if spec else f.cls)

        # Findings above the horizon have no ground fix. Fall back to the camera
        # position and say so, rather than inventing a distance.
        lat, lon, approx = f.lat, f.lon, False
        if lat is None or lon is None:
            fr = (
                next((x for x in frames if x.frame_id == det.frame_id), None)
                if det
                else None
            )
            if fr is None:
                skipped += 1
                continue
            lat, lon, approx = fr.lat, fr.lon, True

        img_name = None
        if det is not None:
            jpeg = _evidence_jpeg(
                run_dir / "frames" / f"{det.frame_id}.jpg", det, draw_box=not absent
            )
            if jpeg:
                img_name = f"img/{f.finding_id}.jpg"
                images[img_name] = jpeg

        balloon = _balloon(f, det, img_name, absent, index_name, approx)
        when = wall_time(f.t_sec)
        time_el = f"<TimeStamp><when>{when}</when></TimeStamp>" if when else ""

        placemarks.append(
            "<Placemark>"
            f"<name>{_esc(label)} {SEP} sev {f.severity}</name>"
            f"<styleUrl>#c-{_esc(f.cls)}</styleUrl>"
            f"<description>{_cdata(balloon)}</description>"
            f"{time_el}"
            "<Point><altitudeMode>clampToGround</altitudeMode>"
            f"<coordinates>{_coord(lat, lon)}</coordinates></Point>"
            "</Placemark>"
        )

        # An absence also gets a line spanning the stretch it refers to, because
        # a lone pin would claim a point location that does not exist.
        if absent:
            g = gap_for(f)
            if g is not None:
                pts = " ".join(_coord(a, b) for a, b in gap_polyline(frames, g))
                gap_lines.append(
                    "<Placemark>"
                    f"<name>{_esc(label)} {SEP} {g.length_m:.0f} m</name>"
                    f"<styleUrl>#c-{_esc(f.cls)}</styleUrl>"
                    f"<description>{_cdata(balloon)}</description>"
                    "<LineString><altitudeMode>clampToGround</altitudeMode>"
                    f"<tessellate>1</tessellate><coordinates>{pts}</coordinates>"
                    "</LineString>"
                    "</Placemark>"
                )

    # ---- heatmap ----------------------------------------------------------
    seg_marks = [
        "<Placemark>"
        f"<name>{_esc(index_name)} {s.quality_index:.0f}/100</name>"
        f"<description>{_cdata(f'Segment {s.seg_id} &middot; {s.length_m:.0f} m &middot; {len(s.finding_ids)} finding(s)')}</description>"
        f"<Style><LineStyle><color>{_kml_color(s.color, 220)}</color>"
        "<width>8</width></LineStyle></Style>"
        "<LineString><altitudeMode>clampToGround</altitudeMode><tessellate>1</tessellate>"
        f"<coordinates>{_coord(s.start[0], s.start[1])} {_coord(s.end[0], s.end[1])}</coordinates>"
        "</LineString>"
        "</Placemark>"
        for s in segments
    ]

    # ---- route ------------------------------------------------------------
    route_els: list[str] = []
    if frames:
        line = " ".join(_coord(fr.lat, fr.lon) for fr in frames)
        route_els.append(
            "<Placemark><name>Driven route</name><styleUrl>#route</styleUrl>"
            "<LineString><altitudeMode>clampToGround</altitudeMode>"
            f"<tessellate>1</tessellate><coordinates>{line}</coordinates>"
            "</LineString></Placemark>"
        )
        # gx:Track animates the vehicle along the time slider rather than only
        # drawing a static line.
        stamped = [(fr, _kml_time(fr.ts)) for fr in frames]
        stamped = [(fr, w) for fr, w in stamped if w]
        if stamped:
            whens = "".join(f"<when>{w}</when>" for _, w in stamped)
            coords = "".join(
                f"<gx:coord>{fr.lon:.7f} {fr.lat:.7f} 0</gx:coord>" for fr, _ in stamped
            )
            route_els.append(
                "<Placemark><name>Vehicle</name><styleUrl>#route</styleUrl>"
                "<gx:Track><altitudeMode>clampToGround</altitudeMode>"
                f"{whens}{coords}</gx:Track></Placemark>"
            )

    # ---- tour -------------------------------------------------------------
    tour_el = ""
    if tour and findings:
        # Worst first when choosing WHICH to visit, then route order for the
        # actual flight -- a tour that teleports back and forth is unwatchable.
        stops = sorted(
            (f for f in findings if f.lat is not None and f.lon is not None),
            key=lambda x: (-x.severity, x.t_sec),
        )[:max_tour_stops]
        stops.sort(key=lambda x: x.t_sec)

        items = []
        if frames:
            items.append(
                "<gx:FlyTo><gx:duration>3.0</gx:duration>"
                "<gx:flyToMode>bounce</gx:flyToMode>"
                f"{_look_at(frames[0].lat, frames[0].lon, 400, 55)}</gx:FlyTo>"
            )
        for f in stops:
            items.append(
                "<gx:FlyTo><gx:duration>2.2</gx:duration>"
                "<gx:flyToMode>smooth</gx:flyToMode>"
                f"{_look_at(f.lat, f.lon, 110)}</gx:FlyTo>"
                "<gx:Wait><gx:duration>1.0</gx:duration></gx:Wait>"
            )
        tour_el = (
            "<gx:Tour><name>Inspection flythrough</name>"
            f"<gx:Playlist>{''.join(items)}</gx:Playlist></gx:Tour>"
        )

    # ---- document ---------------------------------------------------------
    summary = manifest.get("summary") or {}
    origin = manifest.get("origin") or [0.0, 0.0]
    doc_desc = (
        "<div style='font-family:Arial,sans-serif'>"
        f"<b>{_esc(manifest.get('domain_label') or manifest.get('domain'))}</b><br/>"
        f"{_esc(index_name)}: <b>{summary.get('quality_index', '?')}/100</b> "
        f"(grade {summary.get('grade', '?')})<br/>"
        f"{manifest.get('n_findings', 0)} findings from "
        f"{manifest.get('n_detections', 0)} detections over "
        f"{summary.get('route_length_m', 0):.0f} m<br/>"
        f"Perception backend: <b>{_esc(manifest.get('backend'))}</b><br/>"
        f"Created {_esc(manifest.get('created'))}<br/><br/>"
        "<i>Generated by PhysicalOS. Positions come from monocular ground-plane "
        "projection and are accurate to roughly &plusmn;2&ndash;4 m &mdash; not "
        "survey grade. Road and building context in the source run is "
        "OpenStreetMap data, ODbL.</i></div>"
    )

    folders = [
        ("Findings", placemarks),
        ("Missing assets", gap_lines),
        (f"{index_name} heatmap", seg_marks),
        ("Route", route_els),
    ]
    folder_xml = "".join(
        f"<Folder><name>{_esc(name)} ({len(items)})</name>"
        f"<open>{1 if name == 'Findings' else 0}</open>{''.join(items)}</Folder>"
        for name, items in folders
        if items
    )

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2" '
        'xmlns:gx="http://www.google.com/kml/ext/2.2">'
        "<Document>"
        f"<name>PhysicalOS {SEP} {_esc(manifest.get('run_id', 'run'))}</name>"
        f"<description>{_cdata(doc_desc)}</description>"
        f"{_look_at(origin[0], origin[1], 600, 55)}"
        f"{''.join(styles)}{folder_xml}{tour_el}"
        "</Document></kml>"
    )

    # ---- pack -------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        # doc.kml first: Google Earth opens the first .kml entry in the archive.
        z.writestr("doc.kml", kml)
        z.writestr("img/pin.png", _pin_png(hollow=False))
        z.writestr("img/gap.png", _pin_png(hollow=True))
        for name, blob in images.items():
            z.writestr(name, blob)

    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "placemarks": len(placemarks),
        "gap_lines": len(gap_lines),
        "segments": len(seg_marks),
        "images": len(images),
        "tour_stops": tour_el.count("<gx:FlyTo>"),
        "has_track": "<gx:Track>" in kml,
        "skipped": skipped,
    }
