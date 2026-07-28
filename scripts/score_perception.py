#!/usr/bin/env python
"""Measure precision, recall and F1 of a run against the placed ground truth.

WHY THIS IS SEPARATE FROM verify_sample.py
verify_sample.py is a pass/fail CORRECTNESS gate: it asserts every placed object
was recovered and that positions land sub-metre, and exits non-zero otherwise. It
answers "is the geometry right?".

This answers a different question -- "how good is the perception?" -- which is a
measurement, not a gate. Precision and recall have no correct value to assert;
they are numbers to compare between configurations. Combining the two would mean
a recall dip failing the build.

USE
    uv run python scripts/score_perception.py --run runs/e2e
    uv run python scripts/score_perception.py --run runs/e2e --compare runs/e2e_verified

TWO MEASUREMENTS OF TWO DIFFERENT THINGS, BOTH REPORTED
This script prints two scorecards per run, because the pipeline can fail in two
unrelated ways and one number cannot separate them.

  1. GEO-MATCHED FINDINGS (this script's original definition, unchanged).
     Greedy, by class then nearest position, with a distance ceiling -- the same
     scheme verify_sample.py uses, so the two never disagree about what counts as
     a match. Each finding matches at most one object and vice versa. Inputs are
     <run>/findings.json (deduplicated real-world things) and
     samples/road/objects.json (the objects that were PLACED in the scene).

       true positive   a finding matched to a placed object of the same class
       false positive  a finding with no match (wrong class, or nothing there)
       false negative  a placed object that no finding matched

     The ceiling matters: without one, a pothole reported 40 m from the only real
     pothole would score as a hit. 8 m is about twice the stated +/-2-4 m
     accuracy, so it forgives localisation error without forgiving invention.

     This is the number the README publishes (runs/e2e: 63.6% / 48.3% / 54.9) and
     it is the number a client cares about, because a client gets findings on a
     map, not boxes on a frame. It is also the number that CANNOT say why a miss
     happened: a defect the detector never saw and a defect it saw with a box so
     wrong that projection put it 20 m away both land in "false negative".

  2. BOX-LEVEL DETECTION METRICS, computed by supervision.
     sv.MeanAveragePrecision for mAP50 / mAP50-95 / mAP75, and sv.ConfusionMatrix
     for the per-class picture. Inputs are the run's per-frame boxes and
     samples/road/truth.json -- the same fixture `pos perceive --backend mock
     --truth` replays, keyed by keyframe t_sec, boxes normalised 0..1000.
     Matching is IoU in pixels, in-frame, with no geometry and no clustering in
     the way, so a bad box shows up as a bad box.

     Standard implementations rather than more bespoke matching code, and they
     answer what the geo score cannot: WHICH classes are confused, and whether an
     error was a miss, an invention or a mislabel.

WHY BOTH DEFINITIONS ARE KEPT AND LABELLED
supervision's precision/recall (box IoU >= 0.5, counted per sighting) and this
script's precision/recall (position within 8 m, counted per finding) are different
quantities that happen to share a name. On runs/e2e they disagree by more than a
factor of two -- 63.6% geo precision against 25.8% box precision -- because
clustering fuses several badly-boxed sightings into one finding whose position is
still within 8 m of the real defect. Both are true. Silently replacing the
published number with the other one would invalidate every figure quoted in the
README, so the original lines are printed verbatim and the new ones all say
"box".

WHAT THE BOX METRICS CANNOT SEE
`pos verify` rejects FINDINGS, not detections, so runs/e2e and runs/e2e_verified
carry the identical 31 raw detections and score identically under --boxes-from
detections (measured: both mAP50 6.9%). Use --boxes-from findings to score only
the sightings that survived into findings.json (31 against 25 on those two runs)
when the thing being measured is a finding-level filter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pos.config import DomainConfig  # noqa: E402
from pos.geo import haversine_m  # noqa: E402
from pos.schema import BOX_SCALE, Detection  # noqa: E402
from pos.svbridge import SvBridgeError, class_order, to_sv  # noqa: E402

MATCH_M = 8.0

# IoU floor for the confusion matrix. 0.5 is the COCO convention and what
# supervision itself defaults to; mAP50-95 sweeps 0.50..0.95 regardless, so this
# only fixes the single threshold the per-class table is counted at.
BOX_IOU = 0.5

# Confidence floor for the confusion matrix. supervision defaults this to 0.3; we
# default to 0.0 deliberately. The geo scorecard above applies NO confidence
# filter, so a filter here would count fewer predictions than the FP figure
# printed three lines earlier -- two numbers in one report that cannot be
# reconciled. Every backend already applies its own floor upstream
# (pos/perception/onnx_yolo.py uses conf 0.30), so 0.0 drops nothing on real
# runs: the lowest confidence in runs/e2e is 0.80.
BOX_CONF = 0.0


def load(path: Path):
    if not path.exists():
        sys.exit(f"missing: {path}")
    return json.loads(path.read_text())


def match(findings: list[dict], objects: list[dict], max_m: float = MATCH_M):
    """Greedy class+distance matching. Returns (pairs, unmatched_f, unmatched_o)."""
    used_f: set[int] = set()
    pairs: list[tuple[int, int, float]] = []

    for oi, obj in enumerate(objects):
        best, best_d = -1, float("inf")
        for fi, f in enumerate(findings):
            if fi in used_f or f.get("cls") != obj["cls"]:
                continue
            if f.get("lat") is None or f.get("lon") is None:
                continue
            d = haversine_m(obj["lat"], obj["lon"], f["lat"], f["lon"])
            if d < best_d:
                best, best_d = fi, d
        if best >= 0 and best_d <= max_m:
            used_f.add(best)
            pairs.append((best, oi, best_d))

    unmatched_f = [i for i in range(len(findings)) if i not in used_f]
    matched_o = {oi for _, oi, _ in pairs}
    unmatched_o = [i for i in range(len(objects)) if i not in matched_o]
    return pairs, unmatched_f, unmatched_o


def score(findings: list[dict], objects: list[dict], max_m: float = MATCH_M) -> dict:
    pairs, fp, fn = match(findings, objects, max_m)
    tp = len(pairs)
    precision = tp / len(findings) if findings else 0.0
    recall = tp / len(objects) if objects else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    errs = sorted(d for _, _, d in pairs)
    return {
        "findings": len(findings),
        "objects": len(objects),
        "tp": tp,
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_err_m": errs[len(errs) // 2] if errs else None,
        "fp_classes": sorted({findings[i].get("cls", "?") for i in fp}),
        "fn_classes": sorted({objects[i]["cls"] for i in fn}),
    }


def report(name: str, s: dict) -> None:
    print(f"\n  {name}")
    print(f"    findings {s['findings']}   placed objects {s['objects']}")
    print(f"    TP {s['tp']}   FP {s['fp']}   FN {s['fn']}")
    print(
        f"    precision {s['precision'] * 100:5.1f}%   "
        f"recall {s['recall'] * 100:5.1f}%   F1 {s['f1'] * 100:5.1f}"
    )
    if s["median_err_m"] is not None:
        print(f"    median position error {s['median_err_m']:.2f} m")
    if s["fp_classes"]:
        print(f"    false positives in: {', '.join(s['fp_classes'])}")
    if s["fn_classes"]:
        print(f"    missed entirely:    {', '.join(s['fn_classes'])}")


# ---------------------------------------------------------------------------
# Box-level scoring, computed by supervision
# ---------------------------------------------------------------------------


def frame_dims(frames: list[dict]) -> dict[str, tuple[int, int]]:
    """frame_id -> (width, height) in pixels, read PER FRAME.

    Per frame and not frames[0]: pos/svbridge.py cannot detect a wrong frame size
    -- the 0..1000 form has thrown the pixel dimensions away -- so a run that
    mixed resolutions would silently misplace every box from the odd frame with no
    error anywhere. See "WHAT HAPPENS IF THE FRAME SIZE IS WRONG" in that module.
    """
    dims: dict[str, tuple[int, int]] = {}
    for f in frames:
        w, h = int(f["width"]), int(f["height"])
        if w <= 0 or h <= 0:
            sys.exit(f"frame {f['frame_id']} has non-positive size {w}x{h}")
        dims[f["frame_id"]] = (w, h)
    return dims


def probe_bridge(wh: tuple[int, int], domain: DomainConfig) -> None:
    """Assert to_sv maps the full 0..BOX_SCALE box onto exactly this frame.

    The full-frame box is the one case whose correct answer is known without
    reference to an image, so this catches a transposed or inverted scale -- the
    failure that comparing the two sides' dimensions cannot see, because ground
    truth and predictions would then be wrong together and IoU would still look
    healthy.
    """
    key = class_order(domain)[0]
    probe = Detection(
        frame_id="_probe",
        cls=key,
        box=[0.0, 0.0, BOX_SCALE, BOX_SCALE],
        severity=1,
        confidence=1.0,
    )
    x1, y1, x2, y2 = (float(v) for v in to_sv([probe], wh[0], wh[1], domain).xyxy[0])
    if (x1, y1) != (0.0, 0.0) or (round(x2), round(y2)) != wh:
        sys.exit(
            f"pos.svbridge.to_sv maps the full frame to [{x1},{y1},{x2},{y2}] on a "
            f"{wh[0]}x{wh[1]} frame, expected [0,0,{wh[0]},{wh[1]}]. The box "
            "convention has changed underneath this script; IoU against the truth "
            "fixture would be meaningless."
        )


def read_predictions(run: Path, source: str) -> dict[str, list[Detection]]:
    """frame_id -> that frame's predicted Detections, from one of two sources.

    "detections" is raw perception output: everything the detector said, before
    clustering ran. "findings" is the subset kept as evidence in findings.json,
    which is what the run actually reports. They differ exactly when a
    finding-level stage drops something -- `pos verify` removes 6 of runs/e2e's 31
    sightings that way -- and detections.ndjson cannot show it, because verify
    never rewrites that file.
    """
    if source == "detections":
        lines = (run / "detections.ndjson").read_text().splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    elif source == "findings":
        rows = [ev for f in load(run / "findings.json") for ev in f.get("evidence", [])]
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(source)

    by_frame: dict[str, list[Detection]] = {}
    for row in rows:
        det = Detection(**row)
        by_frame.setdefault(det.frame_id, []).append(det)
    return by_frame


def truth_for_frame(
    items: list[dict], frame_id: str, t_key: str
) -> list[Detection]:
    """One truth.json entry -> our Detections, with the convention checked.

    A fixture holding PIXELS instead of the normalised form is the failure worth
    catching here: it would convert without complaint (svbridge only refuses
    non-positive frame sizes) and produce boxes far outside the frame, so every
    IoU would be 0 and the run would look like a total detector failure rather
    than a bad fixture. Coordinates above BOX_SCALE are the signature of that
    mistake, so refuse instead.
    """
    out: list[Detection] = []
    for item in items:
        box = [float(v) for v in item["box"]]
        if max(box) > BOX_SCALE * 1.001 or min(box) < -BOX_SCALE * 0.001:
            sys.exit(
                f"truth box {box} at t={t_key} lies outside 0..{BOX_SCALE:.0f}. The "
                "fixture appears to hold PIXELS, not the normalised convention "
                "pos/schema.py defines. Rebuild it with scripts/make_truth.py build."
            )
        out.append(
            Detection(
                frame_id=frame_id,
                cls=item["cls"],
                box=box,
                severity=int(item.get("severity", 3)),
                # The fixture carries a confidence because it doubles as a
                # mock-backend replay file. It is not a belief about the truth,
                # and supervision ignores confidence on the target side.
                confidence=float(item.get("confidence", 1.0)),
            )
        )
    return out


def build_box_pairs(
    run: Path,
    truth_boxes: dict[str, list[dict]],
    domain: DomainConfig,
    source: str,
) -> dict[str, Any]:
    """Per-frame (prediction, ground truth) sv.Detections pairs for one run.

    THE FRAME SIZE IS LOOKED UP ONCE PER FRAME AND USED FOR BOTH SIDES. Both files
    store boxes normalised to 0..1000, so each side is scaled to pixels by that
    frame's own width/height; converting the two sides with different dimensions
    would scale them apart and make every IoU wrong while every individual box
    still looked plausible. The dimensions actually passed are recorded per side
    and compared afterwards, so a later refactor that splits the lookup fails
    loudly instead of quietly reporting a worse model.
    """
    frames = load(run / "frames.json")
    dims = frame_dims(frames)
    for wh in sorted(set(dims.values())):
        probe_bridge(wh, domain)

    by_frame = read_predictions(run, source)
    orphans = sorted(set(by_frame) - set(dims))

    preds: list[sv.Detections] = []
    targets: list[sv.Detections] = []
    pred_dims: dict[str, tuple[int, int]] = {}
    gt_dims: dict[str, tuple[int, int]] = {}
    unknown_truth: list[str] = []

    for f in frames:
        # truth.json is keyed by the keyframe's t_sec formatted to 2 dp --
        # scripts/make_truth.py:268 writes f"{f['t_sec']:.2f}". A frame whose key
        # is absent has UNKNOWN ground truth, which is not the same as EMPTY
        # ground truth: scoring it as empty would turn every prediction on it into
        # a false positive against a fixture that never made a claim there. So
        # those frames are excluded from both lists and counted.
        t_key = f"{f['t_sec']:.2f}"
        fid = f["frame_id"]
        if t_key not in truth_boxes:
            unknown_truth.append(fid)
            continue

        wh = dims[fid]
        gt = truth_for_frame(truth_boxes[t_key], fid, t_key)
        try:
            pred_sv = to_sv(by_frame.get(fid, []), wh[0], wh[1], domain)
            gt_sv = to_sv(gt, wh[0], wh[1], domain)
        except SvBridgeError as exc:
            sys.exit(f"frame {fid} does not fit domain {domain.key!r}: {exc}")

        pred_dims[fid] = wh
        gt_dims[fid] = wh
        preds.append(pred_sv)
        targets.append(gt_sv)

    if pred_dims != gt_dims:
        differing = sorted(k for k in pred_dims if pred_dims[k] != gt_dims.get(k))
        sys.exit(
            "ground truth and predictions were converted with different frame "
            f"dimensions on {len(differing)} frame(s), e.g. {differing[:3]}. IoU "
            "matching would be meaningless."
        )

    return {
        "preds": preds,
        "targets": targets,
        "n_frames": len(preds),
        "n_gt": int(sum(len(t) for t in targets)),
        "n_pred": int(sum(len(p) for p in preds)),
        "source": source,
        "frames_unknown_truth": unknown_truth,
        "orphan_frame_ids": orphans,
    }


def box_metrics(
    pairs: dict[str, Any],
    domain: DomainConfig,
    iou: float = BOX_IOU,
    conf: float = BOX_CONF,
) -> dict:
    """mAP and a per-class confusion picture, both computed by supervision."""
    preds, targets = pairs["preds"], pairs["targets"]
    classes = class_order(domain)

    out: dict[str, Any] = {k: v for k, v in pairs.items() if k not in ("preds", "targets")}
    out["iou"] = iou
    out["conf"] = conf
    out["per_class"] = []
    out["confusions"] = []

    if not preds or (pairs["n_gt"] == 0 and pairs["n_pred"] == 0):
        # mAP over an empty comparison is undefined, not zero. Printing 0.0 would
        # read as "the detector failed" when the truth was that nothing was asked.
        out["map"] = None
        return out

    result = MeanAveragePrecision().update(preds, targets).compute()
    ap50_by_class = {
        int(cid): float(result.ap_per_class[i][0])
        for i, cid in enumerate(result.matched_classes)
    }
    out["map"] = {
        "map50": float(result.map50),
        "map75": float(result.map75),
        "map50_95": float(result.map50_95),
        # COCO size buckets, by ground-truth box AREA IN PIXELS (small < 32^2,
        # medium < 96^2). On a 1280x720 frame a "small" box is under 0.11% of the
        # frame, so road defects land almost entirely in "large" and the other
        # buckets are usually absent -- shown as n/a rather than 0.0, which would
        # read as a failure on sizes that were never present.
        "small": _bucket_map(result.small_objects),
        "medium": _bucket_map(result.medium_objects),
        "large": _bucket_map(result.large_objects),
    }

    cm = sv.ConfusionMatrix.from_detections(
        predictions=preds,
        targets=targets,
        classes=classes,
        conf_threshold=conf,
        iou_threshold=iou,
    )
    m = np.asarray(cm.matrix, dtype=float)
    n = len(classes)
    # ORIENTATION, pinned at the point of use because a transposed confusion
    # matrix still looks like a confusion matrix: supervision indexes
    # matrix[true_class, predicted_class], and it reserves index n on BOTH axes
    # for "nothing". So m[i, n] is ground truth that no prediction overlapped (a
    # miss) and m[n, j] is a prediction that overlapped no ground truth (an
    # invention). Verified against supervision's own docstring example in
    # ConfusionMatrix.from_tensors, whose duplicate person box lands in m[n, 0].
    tp_total = float(np.trace(m[:n, :n]))
    pred_total = float(m[:, :n].sum())
    gt_total = float(m[:n, :].sum())
    precision = tp_total / pred_total if pred_total else 0.0
    recall = tp_total / gt_total if gt_total else 0.0
    out["box_tp"] = int(round(tp_total))
    out["box_fp"] = int(round(pred_total - tp_total))
    out["box_fn"] = int(round(gt_total - tp_total))
    out["box_precision"] = precision
    out["box_recall"] = recall
    out["box_f1"] = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    for i, key in enumerate(classes):
        gt_i = float(m[i, :].sum())
        pred_i = float(m[:, i].sum())
        if not gt_i and not pred_i:
            # A 15-class domain against a 9-class scene would otherwise print six
            # rows of zeros and bury the six that carry the answer.
            continue
        out["per_class"].append(
            {
                "cls": key,
                "gt": int(round(gt_i)),
                "pred": int(round(pred_i)),
                "tp": int(round(m[i, i])),
                "fp": int(round(pred_i - m[i, i])),
                "fn": int(round(gt_i - m[i, i])),
                # The split the geo scorecard cannot make: a defect nothing
                # overlapped at all, versus one a prediction overlapped and then
                # called something else. The first is a detector blind spot, the
                # second is a taxonomy problem, and they need different fixes.
                "missed": int(round(m[i, n])),
                "invented": int(round(m[n, i])),
                "ap50": ap50_by_class.get(i),
            }
        )

    out["confusions"] = sorted(
        (
            (classes[i], classes[j], int(round(m[i, j])))
            for i in range(n)
            for j in range(n)
            if i != j and m[i, j]
        ),
        key=lambda row: -row[2],
    )
    return out


def _bucket_map(bucket: Any) -> float | None:
    """mAP50-95 of one COCO size bucket, or None when it held no ground truth."""
    if bucket is None:
        return None
    value = float(bucket.map50_95)
    # supervision returns -1.0 for a bucket with nothing of that size in it.
    return None if value < 0 else value


def _pct(value: float | None) -> str:
    return "  n/a " if value is None else f"{value * 100:5.1f}%"


def report_boxes(m: dict, truth_name: str) -> None:
    source = "detections.ndjson" if m["source"] == "detections" else "findings.json evidence"
    print(f"    -- box level vs {truth_name}, computed by supervision --")
    print(
        f"    frames scored {m['n_frames']}   truth boxes {m['n_gt']}   "
        f"predicted boxes {m['n_pred']}   from {source}"
    )
    if m["frames_unknown_truth"]:
        print(
            f"    excluded {len(m['frames_unknown_truth'])} frame(s) whose t_sec has "
            "no truth entry (unknown, not empty)"
        )
    if m["orphan_frame_ids"]:
        print(
            f"    ignored {len(m['orphan_frame_ids'])} predicted frame_id(s) absent "
            f"from frames.json: {', '.join(m['orphan_frame_ids'][:4])}"
        )
    if m["map"] is None:
        print("    mAP undefined: no boxes on either side")
        return

    mp = m["map"]
    print(
        f"    mAP50 {_pct(mp['map50'])}   mAP50-95 {_pct(mp['map50_95'])}   "
        f"mAP75 {_pct(mp['map75'])}"
    )
    print(
        f"    mAP50-95 by box area: small {_pct(mp['small'])}   "
        f"medium {_pct(mp['medium'])}   large {_pct(mp['large'])}"
    )
    print(
        f"    at IoU>={m['iou']:.2f}, conf>={m['conf']:.2f}:   box TP {m['box_tp']}   "
        f"box FP {m['box_fp']}   box FN {m['box_fn']}"
    )
    # The two blocks answer at DIFFERENT thresholds and the numbers below will
    # look inconsistent with the mAP above unless that is said out loud. mAP is
    # defined over the whole precision-recall curve, so it neither takes a
    # confidence floor nor moves when --box-iou changes; the counts and the AP50
    # column do. Only printed when a flag has actually moved off the default,
    # because on a default run there is nothing to reconcile.
    if m["conf"] > 0 or m["iou"] != BOX_IOU:
        print(
            f"    (mAP above is unaffected by --box-conf/--box-iou by definition; "
            f"the counts are at IoU>={m['iou']:.2f} and AP50 stays at IoU>=0.50)"
        )
    # Every one of these says "box" because they are per-sighting IoU numbers and
    # the four lines above them are per-finding 8 m numbers. Same words, different
    # quantities -- see the module docstring.
    print(
        f"    box precision {m['box_precision'] * 100:5.1f}%   "
        f"box recall {m['box_recall'] * 100:5.1f}%   "
        f"box F1 {m['box_f1'] * 100:5.1f}"
    )

    if m["per_class"]:
        print(
            f"    {'per class':<22}{'GT':>5}{'pred':>6}{'TP':>5}{'FP':>5}{'FN':>5}"
            f"{'missed':>8}{'invented':>10}{'AP50':>9}"
        )
        for row in m["per_class"]:
            ap = "        -" if row["ap50"] is None else f"{row['ap50'] * 100:8.1f}%"
            print(
                f"      {row['cls']:<20}{row['gt']:>5}{row['pred']:>6}{row['tp']:>5}"
                f"{row['fp']:>5}{row['fn']:>5}{row['missed']:>8}{row['invented']:>10}{ap}"
            )

    if m["confusions"]:
        print(f"    class confusions (true -> predicted, IoU>={m['iou']:.2f}):")
        for true_cls, pred_cls, count in m["confusions"]:
            print(f"      {true_cls} -> {pred_cls}   {count}")
    else:
        print(
            "    class confusions: none -- every box error is a miss or an "
            "invention, not a mislabel"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--compare", type=Path, help="Second run, compared to --run.")
    ap.add_argument("--truth", type=Path, default=REPO / "samples/road/objects.json")
    ap.add_argument(
        "--truth-boxes",
        type=Path,
        help="Per-frame box fixture for the supervision metrics. Defaults to "
        "truth.json beside --truth, because the two describe the same scene and "
        "scoring boxes from one scene against objects from another is silent "
        "nonsense rather than an error.",
    )
    ap.add_argument("--match-m", type=float, default=MATCH_M)
    ap.add_argument("--box-iou", type=float, default=BOX_IOU)
    ap.add_argument("--box-conf", type=float, default=BOX_CONF)
    ap.add_argument(
        "--boxes-from",
        choices=("detections", "findings"),
        default="detections",
        help="Which predictions the box metrics score: every raw sighting, or only "
        "those kept as evidence in findings.json.",
    )
    ap.add_argument(
        "--no-box-metrics",
        action="store_true",
        help="Print only the original geo-matched scorecard.",
    )
    args = ap.parse_args()

    truth_boxes_path = args.truth_boxes or args.truth.parent / "truth.json"

    objects = load(args.truth)
    print("=" * 68)
    print(f"  Perception quality vs {args.truth.name} ({len(objects)} placed objects)")
    print(f"  match ceiling {args.match_m:.0f} m")
    print("=" * 68)

    truth_boxes: dict[str, list[dict]] | None = None
    if not args.no_box_metrics:
        if truth_boxes_path.exists():
            truth_boxes = json.loads(truth_boxes_path.read_text())
        else:
            # Not fatal. The geo scorecard is the published one and stands on its
            # own; a scene with no box fixture simply gets one scorecard, and
            # saying so beats printing mAP 0.0 for a file that does not exist.
            print(
                f"\n  no box fixture at {truth_boxes_path} -- skipping the "
                "supervision metrics"
            )

    def box_for(run: Path) -> dict | None:
        if truth_boxes is None:
            return None
        # The DOMAIN comes from the run's own manifest, not a flag: class_order()
        # in pos/svbridge.py is what defines class_id, so ground truth and
        # predictions must be numbered against the same taxonomy or the confusion
        # matrix would compare two different class lists.
        domain_key = json.loads((run / "manifest.json").read_text())["domain"]
        domain = DomainConfig.load(domain_key)
        pairs = build_box_pairs(run, truth_boxes, domain, args.boxes_from)
        return box_metrics(pairs, domain, args.box_iou, args.box_conf)

    a = score(load(args.run / "findings.json"), objects, args.match_m)
    report(str(args.run), a)
    ma = box_for(args.run)
    if ma:
        report_boxes(ma, truth_boxes_path.name)

    if args.compare:
        b = score(load(args.compare / "findings.json"), objects, args.match_m)
        report(str(args.compare), b)
        mb = box_for(args.compare)
        if mb:
            report_boxes(mb, truth_boxes_path.name)

        print("\n  delta (compare - run)")
        for key, label in (("precision", "precision"), ("recall", "recall"), ("f1", "F1")):
            print(f"    {label:10} {(b[key] - a[key]) * 100:+6.1f}")
        print(f"    {'findings':10} {b['findings'] - a['findings']:+6d}")
        # The whole point of a verification pass: fewer false positives without
        # losing true ones.
        print(
            f"    {'FP':10} {b['fp'] - a['fp']:+6d}"
            f"    TP {b['tp'] - a['tp']:+d}"
        )
        if ma and mb and ma["map"] and mb["map"]:
            for key, label in (("map50", "box mAP50"), ("map50_95", "box mAP50-95")):
                print(f"    {label:12} {(mb['map'][key] - ma['map'][key]) * 100:+6.1f}")
            print(
                f"    {'box prec':12} "
                f"{(mb['box_precision'] - ma['box_precision']) * 100:+6.1f}"
                f"    box recall {(mb['box_recall'] - ma['box_recall']) * 100:+6.1f}"
            )
            if ma["n_pred"] == mb["n_pred"] and args.boxes_from == "detections":
                # Stops a null result being read as a real one. `pos verify` drops
                # findings, so the two detections.ndjson files are identical and
                # every box delta is arithmetically forced to zero -- which is not
                # the same as the verification pass having no effect.
                print(
                    f"    (both runs hold the same {ma['n_pred']} raw detections, so "
                    "the box deltas cannot move -- rerun with --boxes-from findings)"
                )

        v = args.compare / "verification.json"
        if v.exists():
            st = json.loads(v.read_text()).get("stats", {})
            print(
                f"\n  verification: checked {st.get('checked')}, "
                f"confirmed {st.get('confirmed')}, rejected {st.get('rejected')}, "
                f"unsure {st.get('unsure')}"
            )

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
