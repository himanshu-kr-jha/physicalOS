"""Import findings from an external CSV, so a survey someone else scored can be viewed.

WHY THIS EXISTS
Not every finding comes from our detector. Field teams and third-party contractors
deliver defect lists as spreadsheets with a lat/lon per row, and those deserve the
same 3D route, satellite ground and evidence panel as a run we perceived ourselves.
The column contract is exactly what `pos export-csv` writes, so a run round-trips:
export a scored run, hand the CSV out, take the edits back.

WHY EVERY ROW IS PARSED DEFENSIVELY
The CSV comes out of a spreadsheet, which means a stray "N/A" in a lat column and a
"1:2:3:4" in a time column are ordinary, not exceptional. One `float()` raising
halfway through would discard every row already parsed and leave a half-written run
behind, so bad cells are counted and reported rather than thrown.

The counts are printed because a silent skip is how you come to trust a map that is
missing findings -- and because an unparseable time becomes 0.0, which places a
finding at the very start of the route looking entirely plausible.
"""

from __future__ import annotations

import csv
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import typer

from .config import DomainConfig
from .schema import Detection, Finding, Frame, RunManifest

#: Rows carry no bounding box. A zero-size one tells the viewer and the PDF report to
#: show the evidence image without drawing an annotation rectangle over it.
_NO_BOX = [0.0, 0.0, 0.0, 0.0]


def extract_gdrive_id(url: str) -> str:
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else ""


def download_image(url: str, dest: Path) -> bool:
    if not url or not url.startswith("http"):
        return False
    if dest.exists():
        return True

    gdrive_id = extract_gdrive_id(url)
    if gdrive_id:
        url = f"https://drive.google.com/uc?export=download&id={gdrive_id}"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, 'wb') as f:
                    f.write(response.read())
                return True
    except Exception as e:  # noqa: BLE001 - one bad URL must not stop the import
        typer.secho(f"Failed to download {url}: {e}", fg=typer.colors.RED)
    return False


def parse_time(t_str: str) -> float | None:
    """Video time from "M:SS", "H:MM:SS" or bare seconds. None when unreadable.

    None rather than 0.0 so the caller can count the failures. Silently returning
    zero drops the finding at the start of the route, which looks like a real result.
    """
    t_str = (t_str or "").strip()
    if not t_str:
        return None
    try:
        parts = t_str.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 1:
            return float(t_str)
    except ValueError:
        pass
    return None


def _parse_coord(raw: str) -> float | None:
    """One decimal degree, or None when blank or unreadable. Blank is legitimate."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


#: Used when the CSV carries no severity/confidence. A mid severity at full confidence
#: is the least-assuming choice, but it is NOT free: the quality index is
#: weight x severity x confidence, so a file without these columns scores differently
#: from the run it was exported out of. `pos export-csv` writes both for that reason.
DEFAULT_SEVERITY = 3
DEFAULT_CONFIDENCE = 1.0


def _parse_severity(raw: str) -> int | None:
    """An integer 1-5, or None when blank or out of range (schema.Finding's bounds)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        # float() first: a spreadsheet happily writes "3.0" in an integer column.
        v = int(round(float(raw)))
    except ValueError:
        return None
    return v if 1 <= v <= 5 else None


def _parse_confidence(raw: str) -> float | None:
    """A float in 0..1, or None when blank or out of range."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if 0.0 <= v <= 1.0 else None


def _class_resolver(domain_key: str):
    """Build defect_name -> class key, resolved against the domain taxonomy.

    WHY THIS IS NOT JUST A SLUGIFY
    `pos export-csv` writes the human LABEL into defect_name, and plenty of labels
    differ from their key: road_pci maps `hazard` to "Danger Element" and `garbage` to
    "Garbage Accumulation". Slugifying the label invents `danger_element`, a class the
    taxonomy has never heard of -- and that costs twice over:

      * the viewer builds its legend and colours from the domain's keys, so an
        invented class draws in fallback amber and cannot be filtered
      * score.py looks up domain.spec(cls).weight, and an unknown key silently takes
        the 1.0 default instead of the class's real weight, moving the quality index

    Both failures are invisible -- the map still looks right. On one real 83-finding
    run this renamed 4 classes across 35 findings and shifted the index 16.6 -> 12.7.

    Matching the label first fixes every CSV, not only round-tripped ones: a field
    team writing "Danger Element" in a spreadsheet gets the same answer. Raw keys are
    accepted too, so a CSV that carries keys still works.
    """
    try:
        dom = DomainConfig.load(domain_key)
    except FileNotFoundError:
        return lambda name: (name.lower().strip().replace(" ", "_"), False)

    lookup: dict[str, str] = {}
    for key, spec in dom.class_map.items():
        # Keys first, in their own pass, so a label that happens to collide with
        # another class's key cannot shadow that key.
        lookup.setdefault(key.lower(), key)
    for key, spec in dom.class_map.items():
        lookup.setdefault(spec.label.lower().strip(), key)

    def resolve(name: str) -> tuple[str, bool]:
        """(class key, recognised by the taxonomy?)."""
        probe = name.lower().strip()
        if probe in lookup:
            return lookup[probe], True
        slug = probe.replace(" ", "_")
        if slug in lookup:
            return lookup[slug], True
        # Unrecognised: keep the slug so the finding still appears, and let the caller
        # say so. Silently inventing a class is exactly the bug described above.
        return slug, False

    return resolve


def run_import_csv(csv_path: Path, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Which taxonomy do these defect names belong to? The manifest knows, because
    # `pos ingest` recorded the --domain the run was started with. Absent (a
    # standalone import), "road" matches the manifest this function would create.
    domain_key = "road"
    if (run_dir / "manifest.json").exists():
        try:
            domain_key = json.loads(
                (run_dir / "manifest.json").read_text()
            ).get("domain") or "road"
        except (OSError, ValueError):
            pass
    resolve_class = _class_resolver(domain_key)

    existing_frames: list[Frame] = []
    if (run_dir / "frames.json").exists():
        try:
            existing_frames = [
                Frame(**f) for f in json.loads((run_dir / "frames.json").read_text())
            ]
        except Exception as e:  # noqa: BLE001 - fall back to the CSV's own points
            typer.secho(f"Could not read existing frames.json: {e}", fg=typer.colors.RED)
    existing_frames.sort(key=lambda f: f.t_sec)

    findings: list[Finding] = []
    sparse_frames: list[Frame] = []
    seen_ids: set[str] = set()

    # Counted, then reported. Each of these is a row that is not on the map, or is on
    # it somewhere we had to guess.
    n_skipped_blank = 0
    n_bad_time = 0
    n_bad_coord = 0
    n_renamed = 0
    n_default_severity = 0
    #: defect_name -> count, for names the domain taxonomy does not recognise. Those
    #: findings still appear, but in fallback colour and at the default score weight.
    unknown_classes: dict[str, int] = {}

    # utf-8-sig: Excel writes a BOM, which otherwise lands in the first header name
    # and makes "object_id" unfindable for every row in the file.
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("object_id") and not row.get("defect_name"):
                n_skipped_blank += 1
                continue

            t_sec = parse_time(row.get("time", ""))
            if t_sec is None:
                n_bad_time += 1
                t_sec = 0.0

            raw_lat, raw_lon = row.get("lat", ""), row.get("lon", "")
            lat, lon = _parse_coord(raw_lat), _parse_coord(raw_lon)
            # Separate "blank, so use the frame's position" from "garbage in the cell".
            if ((raw_lat or "").strip() and lat is None) or (
                (raw_lon or "").strip() and lon is None
            ):
                n_bad_coord += 1

            defect_name = (row.get("defect_name") or "Unknown").strip()
            cls_key, known = resolve_class(defect_name)
            if not known:
                unknown_classes[defect_name] = unknown_classes.get(defect_name, 0) + 1

            # Optional columns. Absent, the defaults apply and the score is reported
            # as approximate at the end -- a uniform severity is a real difference in
            # the index, not a rounding detail.
            severity = _parse_severity(row.get("severity", ""))
            confidence = _parse_confidence(row.get("confidence", ""))
            if severity is None:
                severity = DEFAULT_SEVERITY
                n_default_severity += 1
            if confidence is None:
                confidence = DEFAULT_CONFIDENCE

            # With an ingested run present, tie the row to the nearest keyframe in
            # time so its evidence image and its place on the timeline agree.
            matched_frame = None
            if existing_frames:
                matched_frame = min(
                    existing_frames, key=lambda fr: abs(fr.t_sec - t_sec)
                )

            if matched_frame:
                frame_id = matched_frame.frame_id
                # The CSV's own position wins where it has one. The frame's is the
                # fallback, which at least puts the finding on the route.
                lat = matched_frame.lat if lat is None else lat
                lon = matched_frame.lon if lon is None else lon
            else:
                frame_id = (row.get("frame") or "").strip()
                if not frame_id:
                    frame_id = f"f_{row.get('object_id', '0')}"

                # No local keyframe to point at, so fetch the image the CSV names.
                frame_url = (row.get("frame_url") or "").strip()
                if frame_url:
                    download_image(frame_url, frames_dir / f"{frame_id}.jpg")

            # Unique ids matter: the viewer keys its reveal set by finding_id
            # (viewer/src/store.ts), so two rows sharing one id collapse into a single
            # marker and the second finding silently disappears.
            finding_id = str(row.get("object_id") or len(findings) + 1).strip()
            if not finding_id or finding_id in seen_ids:
                finding_id = f"{finding_id or 'row'}-{len(findings) + 1}"
                n_renamed += 1
            seen_ids.add(finding_id)

            findings.append(
                Finding(
                    finding_id=finding_id,
                    cls=cls_key,
                    label=defect_name,
                    lat=lat,
                    lon=lon,
                    severity=severity,
                    confidence=confidence,
                    t_sec=t_sec,
                    evidence=[
                        Detection(
                            frame_id=frame_id,
                            cls=cls_key,
                            box=list(_NO_BOX),
                            severity=severity,
                            confidence=confidence,
                            anchor="centre",
                        )
                    ],
                )
            )

            if lat is not None and lon is not None and not existing_frames:
                sparse_frames.append(
                    Frame(
                        frame_id=frame_id,
                        t_sec=t_sec,
                        lat=lat,
                        lon=lon,
                        heading_deg=0.0,
                        path=f"frames/{frame_id}.jpg",
                        width=1920,
                        height=1080,
                    )
                )

    (run_dir / "findings.json").write_text(
        json.dumps([json.loads(f.model_dump_json()) for f in findings], indent=2)
    )

    # With no ingested run, the CSV's own points become the route.
    if not existing_frames and sparse_frames:
        sparse_frames.sort(key=lambda x: x.t_sec)
        (run_dir / "frames.json").write_text(
            json.dumps(
                [json.loads(f.model_dump_json()) for f in sparse_frames], indent=2
            )
        )

    _write_manifest(run_dir, findings, existing_frames or sparse_frames)

    typer.secho(
        f"Imported {len(findings)} findings from {Path(csv_path).name} into {run_dir}",
        fg=typer.colors.GREEN,
    )
    if n_skipped_blank:
        typer.secho(f"  {n_skipped_blank} blank row(s) skipped", fg=typer.colors.YELLOW)
    if n_bad_time:
        typer.secho(
            f"  {n_bad_time} row(s) had an unreadable 'time' -- placed at t=0, which "
            f"puts them at the START of the route. Check the time column format "
            f"(expects M:SS, H:MM:SS or plain seconds).",
            fg=typer.colors.YELLOW,
        )
    if n_bad_coord:
        typer.secho(
            f"  {n_bad_coord} row(s) had an unreadable lat/lon -- "
            + (
                "snapped to the nearest keyframe"
                if existing_frames
                else "left unplaced, so they will not appear on the map"
            ),
            fg=typer.colors.YELLOW,
        )
    if n_renamed:
        typer.secho(
            f"  {n_renamed} duplicate object_id(s) renamed so no finding is lost",
            fg=typer.colors.YELLOW,
        )
    if n_default_severity:
        typer.secho(
            f"  {n_default_severity} row(s) carried no readable 'severity' -- assumed "
            f"{DEFAULT_SEVERITY}. The quality index is weight x severity x confidence, "
            f"so it is INDICATIVE for those, not a reproduction of the original score. "
            f"Add severity (1-5) and confidence (0-1) columns for a faithful index.",
            fg=typer.colors.YELLOW,
        )
    if unknown_classes:
        total = sum(unknown_classes.values())
        typer.secho(
            f"  {total} finding(s) in {len(unknown_classes)} class(es) the "
            f"'{domain_key}' taxonomy does not define. They are on the map, but in "
            f"the fallback colour, absent from the legend filter, and scored at the "
            f"default weight rather than a class-specific one:",
            fg=typer.colors.YELLOW,
        )
        for name, n in sorted(unknown_classes.items(), key=lambda kv: -kv[1]):
            typer.secho(f"    {n:>4}x {name!r}", fg=typer.colors.YELLOW)
        typer.secho(
            f"  Fix by matching the label in configs/domains/{domain_key}.yaml, "
            f"or pick the domain that defines these.",
            fg=typer.colors.YELLOW,
        )


def _write_manifest(
    run_dir: Path, findings: list[Finding], frames: list[Frame]
) -> None:
    """Update the manifest when there is one, create it when there is not.

    Creating it unconditionally is the point. This used to happen only on the
    standalone path, so importing a CSV into a run that had frames but no manifest
    left the viewer with nothing to load -- and it renders a 404 from /api/manifest as
    "No run found", which reads as a broken install rather than a missing file.
    """
    manifest_path = run_dir / "manifest.json"
    duration = max((f.t_sec for f in frames), default=0.0)

    if manifest_path.exists():
        manifest = RunManifest(**json.loads(manifest_path.read_text()))
        manifest.n_detections = len(findings)
        manifest.n_findings = len(findings)
        # Provenance, and not cosmetic. `pos ingest` leaves backend at its "mock"
        # default, which elsewhere in this codebase means SYNTHETIC FIXTURES -- so an
        # imported run would claim in the studio's run table and in the PDF footer to
        # be fake data. It is also what _detectable_assets() reads to decide which
        # absences can be inferred. Say what actually produced these findings.
        manifest.backend = "csv_import"
        # Refreshed as well: an import that supplied the frames must not leave a
        # manifest claiming the run is empty, or the studio's run table reads 0.
        if frames:
            manifest.n_frames = len(frames)
            manifest.duration_sec = max(manifest.duration_sec, duration)
        manifest_path.write_text(json.dumps(manifest.model_dump(), indent=2))
        return

    if not frames:
        # A manifest needs an origin, and inventing one would drop the whole run into
        # the Gulf of Guinea at 0,0 -- a plausible-looking map of nowhere.
        typer.secho(
            "  no manifest written: no row had a usable lat/lon, and there is no "
            "frames.json to borrow a position from. The viewer needs one or the other.",
            fg=typer.colors.RED,
        )
        return

    manifest_path.write_text(
        json.dumps(
            RunManifest(
                run_id=run_dir.name,
                domain="road",
                domain_label="Road",
                created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                video="",
                origin=[frames[0].lat, frames[0].lon],
                n_frames=len(frames),
                n_detections=len(findings),
                n_findings=len(findings),
                duration_sec=duration,
                backend="csv_import",
            ).model_dump(),
            indent=2,
        )
    )
