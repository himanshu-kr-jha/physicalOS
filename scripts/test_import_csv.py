#!/usr/bin/env python
"""Checks for the CSV importer. Non-zero exit on failure. No fixtures, no network.

WHAT THIS IS GUARDING
The CSV comes out of a spreadsheet, so the rows that matter are the malformed ones.
Every check below is a way a real file has of being wrong:

  * a bad lat cell used to raise mid-file and discard every row already parsed
  * a bad time cell silently became 0.0, planting a finding at the start of the route
  * a repeated object_id collapsed two findings into one marker in the viewer
  * a run with frames but no manifest got no manifest written, and the viewer renders
    a 404 from /api/manifest as "No run found" -- indistinguishable from a bad install

Runs in milliseconds against a temp dir. `frame_url` is blank in every fixture row on
purpose, so nothing here touches the network.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pos.import_csv import parse_time, run_import_csv  # noqa: E402
from pos.schema import Frame  # noqa: E402

FAILED: list[str] = []


def chk(cond: bool, label: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILED.append(label)


HEADER = "object_id,frame,time,defect_name,lat,lon,chainage_km,frame_url"
#: What `pos export-csv` writes now. severity/confidence are appended, so the
#: eight-column HEADER above must keep working -- both are exercised below.
HEADER_SEV = HEADER + ",severity,confidence"

# Six rows: four sound, one with an unreadable lat, one with an unreadable time, plus
# a duplicate object_id and a blank line. Coordinates are synthetic Delhi-area values.
ROWS = [
    "1,00014,0:07,Pothole,28.65712340,77.20365500,0.000,",
    "2,00028,0:14,Ravelling,28.65740120,77.20401880,0.041,",
    "3,00043,0:21,Waterlogging,N/A,77.20431000,0.079,",             # bad lat
    "4,00057,not-a-time,Garbage,28.65771000,77.20460000,0.11,",     # bad time
    "2,00062,0:29,Footpath Damaged,28.65780000,77.20470000,0.13,",  # duplicate id
    ",,,,,,,",                                                      # blank row
]


def write_csv(path: Path) -> None:
    path.write_text("\n".join([HEADER, *ROWS]) + "\n")


def read_findings(run: Path) -> list[dict]:
    return json.loads((run / "findings.json").read_text())


def make_frames(n: int) -> list[Frame]:
    return [
        Frame(
            frame_id=f"{i:05d}",
            t_sec=float(i) * 5.0,
            lat=28.6571 + i * 0.0002,
            lon=77.2036 + i * 0.0003,
            heading_deg=45.0,
            path=f"frames/{i:05d}.jpg",
            width=1920,
            height=1080,
        )
        for i in range(n)
    ]


def write_frames(run: Path, frames: list[Frame]) -> None:
    (run / "frames.json").write_text(
        json.dumps([json.loads(f.model_dump_json()) for f in frames])
    )


def check_parse_time() -> None:
    print("\nparse_time")
    chk(parse_time("0:07") == 7.0, "M:SS")
    chk(parse_time("1:02:03") == 3723.0, "H:MM:SS")
    chk(parse_time("12.5") == 12.5, "bare seconds")
    # The refusals are the point: None lets the caller COUNT bad cells. Returning 0.0
    # would place the finding at the start of the route and look like a real answer.
    chk(parse_time("not-a-time") is None, "refuses garbage instead of returning 0.0")
    chk(parse_time("") is None, "refuses empty")
    chk(parse_time("1:2:3:4") is None, "refuses four-part time")


def check_standalone() -> None:
    """CSV alone: its own points become the route, and a manifest must be created."""
    print("\nstandalone import (no pre-existing run)")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "csvrun"
        csv_path = Path(tmp) / "defects.csv"
        write_csv(csv_path)

        run_import_csv(csv_path, run)

        findings = read_findings(run)
        # Five non-blank rows in, five findings out. A row with one bad cell is still a
        # real defect someone recorded -- dropping it silently is the failure mode.
        chk(len(findings) == 5, f"5 findings from 5 non-blank rows (got {len(findings)})")

        ids = [f["finding_id"] for f in findings]
        chk(len(set(ids)) == len(ids), f"finding_ids unique (got {ids})")

        by_id = {f["finding_id"]: f for f in findings}
        chk(by_id["1"]["lat"] == 28.65712340, "good lat preserved exactly")
        chk(by_id["1"]["cls"] == "pothole", "defect_name -> class key")
        chk(by_id["1"]["label"] == "Pothole", "original label kept for display")
        chk(by_id["1"]["t_sec"] == 7.0, "0:07 -> 7.0 s")
        chk(by_id["3"]["lat"] is None, "unreadable lat left unplaced, not zeroed")
        chk(by_id["4"]["t_sec"] == 0.0, "unreadable time falls back to 0.0")

        chk((run / "manifest.json").exists(), "manifest created")
        m = json.loads((run / "manifest.json").read_text())
        chk(m["backend"] == "csv_import", "backend recorded as csv_import")
        chk(m["n_findings"] == 5, f"manifest n_findings == 5 (got {m['n_findings']})")
        # Only the four rows with a usable lat/lon can become route points.
        chk(m["n_frames"] == 4, f"manifest n_frames == 4 (got {m['n_frames']})")
        chk(len(m["origin"]) == 2 and m["origin"][0] != 0, "origin is a real position")
        chk((run / "frames.json").exists(), "frames.json built from the CSV points")


def check_into_ingested_run() -> None:
    """CSV over an ingested run: rows snap to keyframes, manifest must be written."""
    print("\nimport into a run that already has frames")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "ingested"
        run.mkdir()
        frames = make_frames(7)
        write_frames(run, frames)

        csv_path = Path(tmp) / "defects.csv"
        write_csv(csv_path)
        run_import_csv(csv_path, run)

        findings = read_findings(run)
        chk(len(findings) == 5, f"5 findings (got {len(findings)})")

        # THE REGRESSION THIS FILE EXISTS FOR: frames present but manifest absent used
        # to mean no manifest was written at all, and the viewer said "No run found".
        chk(
            (run / "manifest.json").exists(),
            "manifest created even though frames already existed",
        )
        m = json.loads((run / "manifest.json").read_text())
        chk(m["n_frames"] == 7, f"n_frames from the ingested run (got {m['n_frames']})")
        chk(m["duration_sec"] == 30.0, f"duration from frames (got {m['duration_sec']})")

        by_id = {f["finding_id"]: f for f in findings}
        # A blank or unreadable lat borrows the matched keyframe's position, so it
        # lands on the route rather than nowhere.
        chk(by_id["3"]["lat"] is not None, "bad lat snapped to the nearest keyframe")
        # A row that HAS a position keeps it -- snapping would move a surveyed point.
        chk(by_id["1"]["lat"] == 28.65712340, "CSV's own lat wins over the keyframe's")
        # Evidence points at a real local keyframe, so the panel has an image to show.
        ev_frames = {f["evidence"][0]["frame_id"] for f in findings}
        chk(
            ev_frames <= {f.frame_id for f in frames},
            f"evidence references real keyframes (got {sorted(ev_frames)})",
        )
        chk(
            all(f["evidence"][0]["box"] == [0, 0, 0, 0] for f in findings),
            "zero-size box so no annotation rectangle is drawn",
        )


def check_manifest_update_preserves_run() -> None:
    """An existing manifest keeps its domain and video: the CSV must not demote them."""
    print("\nimport preserves an existing manifest's domain and video")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "existing"
        run.mkdir()
        write_frames(run, make_frames(3))
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "existing",
                    "domain": "road_pci",
                    "domain_label": "Road Distress (PCI + Ensemble)",
                    "created": "2026-08-05T09:14:02Z",
                    "video": "uploads/existing/video.mp4",
                    "origin": [28.6571, 77.2036],
                    "n_frames": 3,
                    "n_detections": 0,
                    "n_findings": 0,
                    "duration_sec": 10.0,
                    "backend": "ensemble",
                }
            )
        )

        csv_path = Path(tmp) / "defects.csv"
        write_csv(csv_path)
        run_import_csv(csv_path, run)

        m = json.loads((run / "manifest.json").read_text())
        chk(m["domain"] == "road_pci", "domain untouched")
        # The video path is what turns the viewer's video panel and chase camera on.
        chk(m["video"] == "uploads/existing/video.mp4", "video path untouched")
        chk(m["n_findings"] == 5, f"n_findings refreshed (got {m['n_findings']})")
        chk(m["n_detections"] == 5, f"n_detections refreshed (got {m['n_detections']})")
        # "mock" means SYNTHETIC FIXTURES elsewhere in this codebase, and it is what
        # `pos ingest` leaves behind. An imported run must not claim to be fake data
        # in the run table, the PDF footer, or _detectable_assets().
        chk(m["backend"] == "csv_import", f"backend records the import (got {m['backend']!r})")


def check_no_position_writes_no_manifest() -> None:
    """No usable position anywhere: refuse to invent an origin at 0,0."""
    print("\nrefusal: no row has a position and there are no frames")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "nowhere"
        csv_path = Path(tmp) / "defects.csv"
        csv_path.write_text(
            "\n".join([HEADER, "1,00014,0:07,Pothole,,,0.000,"]) + "\n"
        )

        run_import_csv(csv_path, run)

        chk((run / "findings.json").exists(), "findings still written")
        # An origin of 0,0 would put the run in the Gulf of Guinea and render a
        # plausible-looking map of nowhere. Writing nothing is the honest answer.
        chk(
            not (run / "manifest.json").exists(),
            "no manifest invented from a missing origin",
        )


def check_class_resolved_from_taxonomy() -> None:
    """A human LABEL must resolve to its taxonomy KEY, not to a slug of the label.

    THE REGRESSION: `pos export-csv` writes the label ("Danger Element"), and
    slugifying it invents `danger_element` where the taxonomy says `hazard`. The
    viewer's legend and colours are keyed on the real key, and score.py looks up
    domain.spec(cls).weight -- so an invented class draws in fallback amber, cannot be
    filtered, and is scored at the default weight instead of its own. On a real
    83-finding run this renamed 4 classes across 35 findings, moving the index from
    16.6 to 12.7 with nothing on screen to indicate it.
    """
    print("\nclass keys resolve through the domain taxonomy")
    from pos.config import DomainConfig

    dom = DomainConfig.load("road_pci")
    # Real label/key pairs that actually disagree, read from the YAML rather than
    # hardcoded, so a taxonomy edit cannot silently invalidate this check.
    pairs = [
        (spec.label, key)
        for key, spec in sorted(dom.class_map.items())
        if spec.label.lower().replace(" ", "_") != key
    ][:4]
    chk(len(pairs) >= 2, f"road_pci has labels that differ from keys ({len(pairs)})")

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "taxo"
        run.mkdir()
        write_frames(run, make_frames(len(pairs) + 2))
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "taxo",
                    "domain": "road_pci",
                    "domain_label": "Road Distress (PCI + Ensemble)",
                    "created": "2026-08-05T09:14:02Z",
                    "video": "uploads/taxo/video.mp4",
                    "origin": [28.6571, 77.2036],
                    "n_frames": len(pairs) + 2,
                    "n_detections": 0,
                    "n_findings": 0,
                    "duration_sec": 10.0,
                    "backend": "ensemble",
                }
            )
        )

        rows = [
            f"{i + 1},{i:05d},0:0{i},{label},28.6571{i},77.2036{i},0.0{i},"
            for i, (label, _key) in enumerate(pairs)
        ]
        # A name the taxonomy does not define must still import, and be reported.
        rows.append(f"{len(pairs) + 1},00009,0:09,Zorble Fault,28.65790,77.20390,0.9,")
        csv_path = Path(tmp) / "labels.csv"
        csv_path.write_text("\n".join([HEADER, *rows]) + "\n")

        run_import_csv(csv_path, run)

        findings = read_findings(run)
        got = {f["finding_id"]: f["cls"] for f in findings}
        for i, (label, key) in enumerate(pairs):
            chk(got[str(i + 1)] == key, f"{label!r} -> {key} (got {got[str(i + 1)]!r})")
        # Unrecognised names keep their slug, so a finding is never dropped for being
        # off-taxonomy -- it is reported instead.
        chk(
            got[str(len(pairs) + 1)] == "zorble_fault",
            "unknown label keeps its slug rather than being dropped",
        )
        # The label is preserved verbatim for display, whatever the key resolves to.
        labels = {f["finding_id"]: f["label"] for f in findings}
        chk(labels["1"] == pairs[0][0], "original label kept for display")
        # And a raw KEY in the CSV must still work, for a hand-written file.
        chk(
            _resolves_key("road_pci", pairs[0][1]) == pairs[0][1],
            f"raw key {pairs[0][1]!r} still accepted",
        )


def _resolves_key(domain: str, name: str) -> str:
    from pos.import_csv import _class_resolver

    return _class_resolver(domain)(name)[0]


def _run_with_manifest(tmp: Path, rows: list[str], header: str = HEADER) -> list[dict]:
    """Import `rows` into a fresh run that already has frames + a road_pci manifest."""
    run = tmp / "sev"
    run.mkdir()
    write_frames(run, make_frames(len(rows) + 1))
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "sev",
                "domain": "road_pci",
                "domain_label": "Road Distress (PCI + Ensemble)",
                "created": "2026-08-05T09:14:02Z",
                "video": "uploads/sev/video.mp4",
                "origin": [28.6571, 77.2036],
                "n_frames": len(rows) + 1,
                "n_detections": 0,
                "n_findings": 0,
                "duration_sec": 10.0,
                "backend": "ensemble",
            }
        )
    )
    csv_path = tmp / "sev.csv"
    csv_path.write_text("\n".join([header, *rows]) + "\n")
    run_import_csv(csv_path, run)
    return read_findings(run)


def check_severity_confidence_columns() -> None:
    """severity/confidence must be read when present and defaulted when absent.

    WHY IT MATTERS: score.py computes weight x severity x confidence, so importing
    every finding at a uniform severity 3 / confidence 1.0 yields a different quality
    index from the run the CSV came from -- on one real 83-finding run, 6.0 against the
    source's 16.6. The columns are optional so older eight-column files keep working,
    but when they ARE there they must be honoured exactly.
    """
    print("\nseverity + confidence columns")
    from pos.import_csv import DEFAULT_SEVERITY

    with tempfile.TemporaryDirectory() as tmp:
        got = _run_with_manifest(
            Path(tmp),
            [
                "1,00000,0:00,Pothole,28.65710,77.20360,0.00,,5,0.750",
                "2,00001,0:05,Pothole,28.65712,77.20362,0.01,,1,0.200",
                # Out-of-range and unreadable values must fall back, not raise.
                "3,00002,0:10,Pothole,28.65714,77.20364,0.02,,9,0.500",
                "4,00003,0:15,Pothole,28.65716,77.20366,0.03,,3,bogus",
            ],
            header=HEADER_SEV,
        )
        by = {f["finding_id"]: f for f in got}
        chk(by["1"]["severity"] == 5, f"severity 5 read (got {by['1']['severity']})")
        chk(
            abs(by["1"]["confidence"] - 0.75) < 1e-9,
            f"confidence 0.750 read (got {by['1']['confidence']})",
        )
        chk(by["2"]["severity"] == 1, "severity 1 read")
        chk(
            by["3"]["severity"] == DEFAULT_SEVERITY,
            f"out-of-range severity 9 -> default {DEFAULT_SEVERITY}",
        )
        chk(
            by["4"]["confidence"] == 1.0,
            f"unreadable confidence -> 1.0 (got {by['4']['confidence']})",
        )
        # Evidence must agree with the finding, or the report and the map disagree.
        chk(
            by["1"]["evidence"][0]["severity"] == 5,
            "evidence severity matches the finding",
        )

    # The original eight-column contract must still import unchanged.
    with tempfile.TemporaryDirectory() as tmp:
        got = _run_with_manifest(
            Path(tmp), ["1,00000,0:00,Pothole,28.65710,77.20360,0.00,"]
        )
        chk(len(got) == 1, "eight-column CSV still imports")
        chk(
            got[0]["severity"] == DEFAULT_SEVERITY and got[0]["confidence"] == 1.0,
            "eight-column CSV falls back to the documented defaults",
        )


def main() -> int:
    print("=" * 64)
    print("  CSV importer checks")
    print("=" * 64)
    check_parse_time()
    check_standalone()
    check_into_ingested_run()
    check_manifest_update_preserves_run()
    check_no_position_writes_no_manifest()
    check_class_resolved_from_taxonomy()
    check_severity_confidence_columns()

    print("\n" + "=" * 64)
    if FAILED:
        print(f"  {len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("  All CSV importer checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
