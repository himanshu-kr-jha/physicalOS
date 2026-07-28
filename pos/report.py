"""A paginated PDF inspection report.

WHY
The dashboard and the KMZ are both interactive. An engineer who has to act on this
needs something that survives being emailed, printed and filed: a fixed document
with a page per defect carrying its photograph, its coordinates and the model's
stated reason. That is the artefact a road authority actually consumes.

THE ONE NON-OBVIOUS REQUIREMENT
Every page footer names the perception backend. A reader must be able to tell VLM
output from local-YOLO output from `[SYNTHETIC FIXTURE]` mock data. A report that
looks authoritative while concealing which model produced it is worse than no
report -- it launders a guess into a record. Same reasoning puts the +/-2-4 m
accuracy and the absence-inference caveat on the cover rather than in a footnote.

Layout uses the reportlab canvas directly rather than platypus flowables: every
page here is a fixed template -- cover, segment table, then one page per finding --
so a flowable document model would add indirection and buy nothing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from .config import DomainConfig
from .schema import BOX_SCALE, Detection, Finding, Segment

PAGE_W, PAGE_H = 595.28, 841.89  # A4 portrait, points
M = 42.0  # margin

INK = (0.10, 0.11, 0.13)
MUTED = (0.42, 0.45, 0.50)
RULE = (0.80, 0.82, 0.85)


def _grade_rgb(grade: str) -> tuple[float, float, float]:
    return {
        "A": (0.20, 0.60, 0.24),
        "B": (0.45, 0.66, 0.15),
        "C": (0.85, 0.65, 0.05),
        "D": (0.90, 0.45, 0.05),
        "E": (0.85, 0.25, 0.10),
        "F": (0.68, 0.11, 0.11),
    }.get((grade or "").upper(), MUTED)


def _hex_rgb(hex_color: str) -> tuple[float, float, float]:
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return MUTED
    try:
        return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return MUTED


def _best_sighting(evidence: list[Detection]) -> Detection | None:
    """Largest box: the closest, most legible pass. Matches the viewer and the KMZ."""
    if not evidence:
        return None
    return max(evidence, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))


def _boxed_image(frame_path: Path, det: Detection, draw_box: bool, width_px: int = 1100):
    """Evidence frame with the box burned in, as a reportlab ImageReader."""
    if not frame_path.exists():
        return None
    from PIL import Image, ImageDraw
    from reportlab.lib.utils import ImageReader

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
                # Dark stroke under bright: a single bright rectangle disappears
                # against pale tarmac.
                d.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=7)
                d.rectangle([x1, y1, x2, y2], outline=(255, 200, 40), width=4)
            if w > width_px:
                img = img.resize(
                    (width_px, max(1, round(h * width_px / w))), Image.LANCZOS
                )
            buf = BytesIO()
            img.save(buf, "JPEG", quality=85)
            buf.seek(0)
            return ImageReader(buf)
    except OSError:
        return None


def _wrap(
    c,
    text: str,
    x: float,
    y: float,
    width: float,
    leading: float = 12.0,
    font: str = "Helvetica",
    size: float = 9.5,
) -> float:
    """Draw wrapped text. Returns y below the last line."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    c.setFont(font, size)
    line = ""
    for word in (text or "").split():
        trial = f"{line} {word}".strip()
        if stringWidth(trial, font, size) > width and line:
            c.drawString(x, y, line)
            y -= leading
            line = word
        else:
            line = trial
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def build_report(run_dir: Path, out_path: Path, image_width: int = 1100) -> dict:
    """Write the PDF. Returns a summary dict.

    `image_width` caps the embedded photo width in pixels and is the only real
    lever on file size -- every finding page carries one full-width image, so a
    long run grows linearly. Measured on a 173-finding run: 1100 px gives ~232 kB
    per page and a 39 MB file, which exceeds the 25 MB limit most mail providers
    impose. 700 px roughly halves it and is still legible on screen and in print
    at this page width.
    """
    from reportlab.pdfgen import canvas as pdfcanvas

    run_dir = Path(run_dir)
    out_path = Path(out_path)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    findings = [
        Finding(**f) for f in json.loads((run_dir / "findings.json").read_text())
    ]

    def maybe(name: str, default):
        p = run_dir / name
        return json.loads(p.read_text()) if p.exists() else default

    segments = [Segment(**s) for s in maybe("segments.json", [])]
    coverage = maybe("coverage.json", [])
    verification = maybe("verification.json", None)

    try:
        domain = DomainConfig.load(manifest.get("domain", "road"))
    except FileNotFoundError:
        domain = DomainConfig(key="road", label="Road")
    class_map = domain.class_map
    absence_keys = {a.key for a in domain.absences}
    index_name = domain.index_name

    summary = manifest.get("summary") or {}
    backend = str(manifest.get("backend", "?"))
    vmodel = (verification or {}).get("model", "")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = pdfcanvas.Canvas(str(out_path), pagesize=(PAGE_W, PAGE_H))
    c.setTitle(f"PhysicalOS inspection report - {manifest.get('run_id')}")
    page_no = [0]

    def footer() -> None:
        """Provenance on every page -- see the module docstring for why."""
        page_no[0] += 1
        c.setStrokeColorRGB(*RULE)
        c.setLineWidth(0.5)
        c.line(M, 46, PAGE_W - M, 46)
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 7.4)
        prov = f"Perception backend: {backend}"
        if vmodel:
            prov += f"  ·  verification model: {vmodel}"
        if backend == "mock":
            prov += "   *** SYNTHETIC FIXTURES - NOT REAL PERCEPTION ***"
        c.drawString(M, 35, prov)
        c.drawRightString(PAGE_W - M, 35, f"page {page_no[0]}")
        c.drawString(
            M, 25,
            f"PhysicalOS  ·  generated {generated}  ·  "
            "positions ±2–4 m, not survey grade",
        )

    def new_page() -> None:
        footer()
        c.showPage()

    # ------------------------------------------------------------------ cover
    y = PAGE_H - M - 12
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 21)
    c.drawString(M, y, "Road Inspection Report")
    y -= 19
    c.setFont("Helvetica", 11)
    c.setFillColorRGB(*MUTED)
    c.drawString(M, y, str(manifest.get("domain_label") or manifest.get("domain")))
    y -= 30

    grade = str(summary.get("grade", "?"))
    c.setFillColorRGB(*_grade_rgb(grade))
    c.rect(M, y - 54, 96, 66, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 33)
    c.drawCentredString(M + 48, y - 26, grade)
    c.setFont("Helvetica", 8)
    c.drawCentredString(M + 48, y - 45, "GRADE")

    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 27)
    c.drawString(M + 116, y - 18, str(summary.get("quality_index", "?")))
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(*MUTED)
    c.drawString(M + 116, y - 34, f"{index_name} (0-100)")
    y -= 84

    rows = [
        ("Run", str(manifest.get("run_id", ""))),
        (
            "Surveyed",
            f"{summary.get('route_length_m', 0):.0f} m over "
            f"{manifest.get('duration_sec', 0):.0f} s",
        ),
        ("Keyframes", str(manifest.get("n_frames", 0))),
        (
            "Observations",
            f"{manifest.get('n_detections', 0)} detections → {len(findings)} findings",
        ),
        (
            "Origin",
            f"{(manifest.get('origin') or [0, 0])[0]:.5f}, "
            f"{(manifest.get('origin') or [0, 0])[1]:.5f}",
        ),
        ("Captured", str(manifest.get("created", ""))),
    ]
    for k, v in rows:
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 9.5)
        c.drawString(M, y, k)
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(M + 96, y, v)
        y -= 15
    y -= 12

    counts = summary.get("counts") or {}
    if counts:
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(M, y, "Findings by class")
        y -= 16
        for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            if y < 200:
                break
            spec = class_map.get(cls)
            c.setFillColorRGB(*_hex_rgb(spec.color if spec else "#999999"))
            c.rect(M, y - 1, 7, 7, fill=1, stroke=0)
            c.setFillColorRGB(*INK)
            c.setFont("Helvetica", 9.5)
            c.drawString(M + 14, y, spec.label if spec else cls)
            c.drawRightString(M + 210, y, str(n))
            if cls in absence_keys:
                c.setFillColorRGB(*MUTED)
                c.setFont("Helvetica-Oblique", 8)
                c.drawString(M + 220, y, "inferred from coverage, not observed")
            y -= 13
        y -= 10

    if coverage and y > 200:
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(M, y, "Asset coverage")
        y -= 16
        for row in coverage:
            if y < 175:
                break
            found = int(row.get("found", 0))
            ok = found > 0
            c.setFillColorRGB(*((0.20, 0.60, 0.24) if ok else (0.68, 0.11, 0.11)))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(M, y, "OK" if ok else "X")
            c.setFillColorRGB(*INK)
            c.setFont("Helvetica", 9.5)
            c.drawString(M + 20, y, str(row.get("asset_label", row.get("asset", ""))))
            c.setFillColorRGB(*MUTED)
            c.setFont("Helvetica", 9)
            c.drawString(
                M + 200, y,
                f"{found} found · {row.get('per_km', 0)} per km" if ok
                else f"none over {row.get('route_m', 0):.0f} m "
                f"(threshold {row.get('min_gap_m', 0):.0f} m)",
            )
            y -= 13

    if verification:
        st = verification.get("stats", {})
        # A dry run wrote verdicts but did NOT touch findings.json, so the wording
        # has to be conditional -- claiming findings were removed when they are
        # still in the report would misstate what the reader is looking at.
        dry = bool(verification.get("dry_run"))
        if dry:
            note = (
                f"A verification pass examined {st.get('checked', 0)} of "
                f"{st.get('input', 0)} findings WITHOUT applying the result: "
                f"{st.get('confirmed', 0)} confirmed, {st.get('rejected', 0)} would "
                f"be rejected, {st.get('unsure', 0)} inconclusive. Every finding "
                "below is still included and its confidence is unchanged."
            )
        else:
            note = (
                f"A verification pass re-examined {st.get('checked', 0)} of "
                f"{st.get('input', 0)} findings: {st.get('confirmed', 0)} confirmed, "
                f"{st.get('rejected', 0)} rejected and removed, "
                f"{st.get('unsure', 0)} left unconfirmed with reduced confidence."
            )
        c.setFillColorRGB(*MUTED)
        _wrap(
            c, note, M, 158, PAGE_W - 2 * M,
            leading=11, font="Helvetica-Oblique", size=8.5,
        )

    c.setFillColorRGB(*MUTED)
    _wrap(
        c,
        "Method: keyframes are geolocated from the GPS track, defects detected per "
        "frame, then projected onto the road plane using the camera calibration and "
        "clustered so repeat sightings of one object become one finding. Positions "
        "are accurate to roughly ±2–4 m and are not survey grade. Absence findings "
        "(missing lighting, missing footpath) are inferred from a lack of detections "
        "over distance rather than observed directly, and are capped below full "
        "confidence for that reason.",
        M, 122, PAGE_W - 2 * M,
        leading=11, font="Helvetica-Oblique", size=8.5,
    )
    new_page()

    # ---------------------------------------------------------- segment table
    if segments:
        y = PAGE_H - M - 10
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(M, y, f"{index_name} by segment")
        y -= 14
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica-Oblique", 8.5)
        c.drawString(M, y, "Worst first - this is the dispatch order.")
        y -= 20

        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica-Bold", 8.5)
        for label, dx in (
            ("SEG", 0), ("SCORE", 42), ("LENGTH", 100),
            ("FINDINGS", 160), ("START LAT, LON", 230),
        ):
            c.drawString(M + dx, y, label)
        y -= 5
        c.setStrokeColorRGB(*RULE)
        c.setLineWidth(0.5)
        c.line(M, y, PAGE_W - M, y)
        y -= 13

        for s in sorted(segments, key=lambda x: x.quality_index):
            if y < 80:
                new_page()
                y = PAGE_H - M - 10
            c.setFillColorRGB(*INK)
            c.setFont("Helvetica", 9)
            c.drawString(M, y, str(s.seg_id))
            c.setFillColorRGB(*_hex_rgb(s.color))
            c.rect(M + 42, y - 1, 22, 8, fill=1, stroke=0)
            c.setFillColorRGB(*INK)
            c.drawString(M + 70, y, f"{s.quality_index:.0f}")
            c.drawString(M + 100, y, f"{s.length_m:.0f} m")
            c.drawString(M + 160, y, str(len(s.finding_ids)))
            c.setFillColorRGB(*MUTED)
            c.setFont("Helvetica", 8.5)
            c.drawString(M + 230, y, f"{s.start[0]:.5f}, {s.start[1]:.5f}")
            y -= 13
        new_page()

    # ----------------------------------------------------- page per finding
    finding_pages = 0
    for f in sorted(findings, key=lambda x: (-x.severity, x.t_sec)):
        absent = f.cls in absence_keys
        det = _best_sighting(f.evidence)
        spec = class_map.get(f.cls)

        y = PAGE_H - M - 6
        c.setFillColorRGB(*_hex_rgb(spec.color if spec else "#999999"))
        c.rect(M, y - 4, 6, 20, fill=1, stroke=0)
        c.setFillColorRGB(*INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(M + 14, y, f.label or (spec.label if spec else f.cls))
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 9)
        c.drawRightString(PAGE_W - M, y + 2, f.finding_id)
        y -= 19

        c.setFont("Helvetica", 9.5)
        c.setFillColorRGB(*MUTED)
        c.drawString(
            M + 14, y,
            f"severity {f.severity}/5   ·   confidence {f.confidence * 100:.0f}%"
            f"   ·   {len(f.evidence)} sighting(s)",
        )
        y -= 18

        if absent:
            c.setFillColorRGB(0.25, 0.25, 0.28)
            c.rect(M, y - 24, PAGE_W - 2 * M, 32, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(M + 8, y - 4, "NOT DETECTED ANYWHERE ALONG THIS STRETCH")
            c.setFont("Helvetica", 8)
            c.drawString(
                M + 8, y - 17,
                "Inferred from coverage. The photograph is the middle of the gap, "
                "so there is no box to draw.",
            )
            y -= 38

        if det is not None:
            img = _boxed_image(
                run_dir / "frames" / f"{det.frame_id}.jpg",
                det,
                not absent,
                width_px=image_width,
            )
            if img is not None:
                iw, ih = img.getSize()
                draw_w = PAGE_W - 2 * M
                draw_h = draw_w * ih / iw
                if draw_h > y - 240:
                    draw_h = max(60.0, y - 240)
                    draw_w = draw_h * iw / ih
                c.drawImage(
                    img, M, y - draw_h, width=draw_w, height=draw_h,
                    preserveAspectRatio=True, anchor="nw",
                )
                y -= draw_h + 16

            if det.evidence:
                c.setStrokeColorRGB(*RULE)
                c.setLineWidth(2)
                c.line(M, y + 5, M, y - 20)
                c.setFillColorRGB(*INK)
                y = _wrap(
                    c, det.evidence, M + 10, y, PAGE_W - 2 * M - 10,
                    leading=12, font="Helvetica-Oblique", size=9.5,
                )
                y -= 8

        # Per-finding accuracy, not one blanket disclaimer. A triangulated fix
        # has a real residual; a single projection does not, and the reader is
        # entitled to know which they are looking at.
        if f.pos_method == "triangulated" and f.pos_residual_m is not None:
            acc = f"+/-{f.pos_residual_m:.1f} m (triangulated, {f.n_rays} rays)"
        elif f.pos_method == "camera":
            acc = "camera position -- above horizon, never ranged"
        else:
            acc = "+/-2-4 m (single ground-plane projection)"

        facts: list[tuple[str, str]] = [
            (
                "Position",
                f"{f.lat:.6f}, {f.lon:.6f}"
                if f.lat is not None and f.lon is not None
                else "not localised (above horizon)",
            ),
            ("Position accuracy", acc),
            ("Time into drive", f"{f.t_sec:.1f} s"),
        ]
        if det is not None:
            facts += [
                (
                    "Range at sighting",
                    f"{det.range_m:.1f} m ahead"
                    if det.range_m is not None
                    else "above horizon - not ranged",
                ),
                ("Evidence frame", det.frame_id),
                (
                    "Box (0-1000, top-left)",
                    "[" + ", ".join(f"{v:.0f}" for v in det.box) + "]",
                ),
            ]

        for k, v in facts:
            if y < 76:
                break
            c.setFillColorRGB(*MUTED)
            c.setFont("Helvetica", 9)
            c.drawString(M, y, k)
            c.setFillColorRGB(*INK)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(M + 132, y, str(v))
            y -= 13

        new_page()
        finding_pages += 1

    c.save()
    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "pages": page_no[0],
        "findings": len(findings),
        "finding_pages": finding_pages,
        "segments": len(segments),
    }
