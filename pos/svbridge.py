"""Conversion seam between our Detection schema and supervision's sv.Detections.

WHY THIS FILE EXISTS
supervision gives us annotators, trackers, polygon zones, slicers and mAP metrics
for free, but every one of them speaks only sv.Detections. Our schema is the
serialised contract the viewer and the reports depend on, so it cannot change to
suit a library. This module is the single place the two meet: every future
supervision feature imports from here rather than reimplementing the arithmetic,
because the arithmetic is where the silent bugs live.

THE ONLY DIFFERENCE IS SCALE -- pinned here, and again at each point of use
    ours (pos/schema.py):  box = [x1, y1, x2, y2], normalised to 0..BOX_SCALE
                           (1000) on BOTH axes, origin TOP-LEFT, x right, y down.
    supervision:           xyxy = [x1, y1, x2, y2], ABSOLUTE PIXELS,
                           origin TOP-LEFT, x right, y down.
Same corner order, same origin, same axis directions. No transpose, no y-flip, no
cxcywh. The conversion is a per-axis multiply: x by frame_w/BOX_SCALE, y by
frame_h/BOX_SCALE -- and the two factors DIFFER on any frame that is not square,
which is the one thing an "it's just a scale" shortcut gets wrong on the 1920x1080
footage this project runs on.

WHAT HAPPENS IF THE FRAME SIZE IS WRONG
Nothing raises. The 0..1000 form has thrown the pixel dimensions away, so this
module cannot tell a correct size from an incorrect one; it can only refuse
non-positive ones. Consequences, in order of how easy they are to miss:
  * A uniform resize of the source frame (a 960x540 thumbnail of a 1920x1080
    keyframe) is CORRECT and intended -- that is how you annotate a downscaled
    copy.
  * A size of a different aspect ratio stretches the boxes exactly as the image
    itself would stretch, so they still sit on the defect in a stretched image and
    sit wrong on a letterboxed one.
  * Dimensions belonging to a DIFFERENT capture (a portrait clip's size applied to
    dashcam boxes) puts every box in the wrong place with no error at all. Read
    width/height from the frames.json entry FOR THAT frame_id. Do not hardcode,
    and do not reuse frames[0] for a run that mixes resolutions.

WHAT IS DELIBERATELY NOT CARRIED ACROSS
Detection.lat/lon/range_m are derived from the box plus the camera model by
`pos localize`. supervision transforms can legitimately move, merge or drop boxes
(NMS/NMM fusion, tracker smoothing, slicer merges), and a moved box with its old
coordinates still attached is worse than one with none -- it launders a stale
position into the pipeline as if it had been measured. So from_sv leaves them
unset and expects localize to run again. `anchor` goes with them, since it only
means anything alongside a projection. Masks are not bridged either: our
segmentation (road_seg_unet) is frame-level, not per-detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv

from .config import ClassSpec, DomainConfig
from .schema import BOX_SCALE, Detection

# Severity is our own concept -- no supervision annotator, tracker or slicer knows
# about it, so anything that builds an sv.Detections from scratch and hands it to
# from_sv has none to give. 3 is the middle of the schema's 1..5 band: it claims
# nothing either way, rather than inventing a clean bill of health (1) or an
# alarm (5).
DEFAULT_SEVERITY = 3

# Keys we own inside sv.Detections.data. Namespacing them would be tidier, but
# "class_name" is fixed by supervision -- sv.LabelAnnotator and every
# sv.Detections.from_*() loader use exactly that string -- so the rest match its
# bare style instead of standing out.
DATA_CLASS_NAME = "class_name"  # ClassSpec.label: human text, for annotators
DATA_CLS = "cls"                # raw taxonomy key: the round-trip source of truth
DATA_SEVERITY = "severity"      # int 1..5
DATA_FRAME_ID = "frame_id"      # so a merge across frames stays attributable
DATA_EVIDENCE = "evidence"      # detector prose, kept so a round trip through an
                                # annotator does not blank the viewer's evidence
                                # panel


class SvBridgeError(ValueError):
    """Raised when a conversion cannot be completed without guessing."""


def class_order(domain: DomainConfig) -> list[str]:
    """The class keys in the order that DEFINES class_id, lowest index first.

    class_id is an index, and an index means nothing without a fixed order. That
    order is domain.class_map's insertion order: the YAML's `classes` list as
    written, then the synthesised absence classes in `absence` order (see
    DomainConfig.class_map). Two consequences worth knowing before editing a
    domain YAML:

      * REORDERING OR REMOVING A CLASS RENUMBERS EVERY CLASS AFTER IT. class_id
        is safe within one process against one loaded domain, and is NOT safe to
        persist or to compare across config versions. Persist `cls` -- the key is
        what our schema and the viewer already store.
      * Absence classes are included on purpose: pos/absence.py:224 emits real
        Detection rows for them (full-frame box), so they must bridge like
        anything else.
    """
    return list(domain.class_map)


def class_index(domain: DomainConfig) -> dict[str, int]:
    """cls key -> class_id. The inverse of class_order()."""
    return {key: i for i, key in enumerate(class_order(domain))}


def class_palette(domain: DomainConfig) -> sv.ColorPalette:
    """Per-class colours in class_order(), so palette.by_idx(class_id) is right.

    The 3D viewer colours a defect by ClassSpec.color. An annotated frame that
    disagrees with the viewer makes a reviewer distrust both, so both read the
    same YAML field.

    ORDER IS THE CONTRACT: supervision indexes a ColorPalette by class_id when an
    annotator runs with color_lookup=sv.ColorLookup.CLASS, so this list must be in
    class_order() and must cover it completely -- sv.ColorPalette.by_idx wraps
    modulo the palette length, so a short palette mislabels colours instead of
    failing.
    """
    colors: list[sv.Color] = []
    for key in class_order(domain):
        spec: ClassSpec = domain.class_map[key]
        try:
            colors.append(sv.Color.from_hex(spec.color))
        except ValueError as exc:
            # Fixing the YAML is cheap. A silently substituted colour ships to a
            # client's report and gets blamed on the viewer.
            raise SvBridgeError(
                f"domain {domain.key!r} class {key!r} has unparseable "
                f"color {spec.color!r}: {exc}"
            ) from exc
    return sv.ColorPalette(colors)


def to_sv(
    dets: list[Detection],
    frame_w: int,
    frame_h: int,
    domain: DomainConfig,
) -> sv.Detections:
    """Our Detections for one frame -> sv.Detections in that frame's pixels.

    frame_w/frame_h must be the dimensions of the image the boxes were measured
    on, or of a uniform resize of it. See the module docstring for what goes
    wrong otherwise, and how quietly.
    """
    if frame_w <= 0 or frame_h <= 0:
        raise SvBridgeError(f"frame size must be positive, got {frame_w}x{frame_h}")

    # sv.Detections.empty() rather than a hand-built zero-row container, because
    # supervision compares against it by value in places and callers test with
    # is_empty(). CAVEAT FOR CALLERS: empty() carries data={}, so d["class_name"]
    # raises KeyError on it. Guard with `if d.is_empty()` before zipping data
    # columns -- on this footage an empty frame is the common case, not an edge
    # case: most keyframes of a road contain no distress at all.
    if not dets:
        return sv.Detections.empty()

    index = class_index(domain)
    unknown = sorted({d.cls for d in dets} - index.keys())
    if unknown:
        # Refusing beats inventing an id. An id invented here would depend on
        # which detections happened to be in the batch, so the same defect would
        # take a different colour in two frames of one run -- a bug that presents
        # as a rendering glitch and costs a day to trace back to this line.
        raise SvBridgeError(
            f"classes {unknown} are not in domain {domain.key!r} "
            f"(known: {sorted(index)}). Add them to the domain YAML, or filter "
            "them out before converting."
        )

    sx = frame_w / BOX_SCALE
    sy = frame_h / BOX_SCALE
    # x by width, y by height. Origin is TOP-LEFT in both systems, so y is NOT
    # flipped here -- flipping it is the classic way to produce boxes that look
    # plausible on a symmetric test image and are wrong on real footage.
    xyxy = np.array(
        [[d.box[0] * sx, d.box[1] * sy, d.box[2] * sx, d.box[3] * sy] for d in dets],
        dtype=np.float32,
    )
    # Not clipped to the frame. A box outside 0..BOX_SCALE means a detector bug
    # and should stay visible as one; clipping would also stop from_sv being an
    # inverse. supervision's annotators clip at draw time anyway.

    # float32 to match what every sv.Detections.from_*() loader produces, so
    # concatenating ours with model output does not promote dtypes underneath
    # supervision. Measured cost of that narrowing, round-tripping all 592
    # detections of runs/POC-1 (1920x1080): worst box error 5.65e-05 in 0..1000
    # units = 1.09e-04 px, worst confidence error 2.48e-08. Far below the
    # ~1 px the detector itself can justify, so precision is not the reason to
    # widen this.
    confidence = np.array([d.confidence for d in dets], dtype=np.float32)
    class_id = np.array([index[d.cls] for d in dets], dtype=int)

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={
            # Label AND key, both: labels are not unique or stable enough to
            # invert (two domains may share "Pothole"), and keys are not readable
            # enough to draw on a frame.
            DATA_CLASS_NAME: [domain.class_map[d.cls].label for d in dets],
            DATA_CLS: [d.cls for d in dets],
            DATA_SEVERITY: np.array([d.severity for d in dets], dtype=int),
            DATA_FRAME_ID: [d.frame_id for d in dets],
            DATA_EVIDENCE: [d.evidence for d in dets],
        },
    )


def from_sv(
    d: sv.Detections,
    frame_id: str,
    frame_w: int,
    frame_h: int,
    domain: DomainConfig,
    evidence: str = "",
) -> list[Detection]:
    """sv.Detections in frame pixels -> our Detections. The inverse of to_sv.

    frame_w/frame_h must be the size of the image d.xyxy was measured on. Pass the
    same pair that went into to_sv and the boxes come back identical apart from
    the float32 storage step.

    `frame_id` is a FALLBACK: data["frame_id"] wins where present, because
    sv.Detections.merge() legitimately concatenates several frames into one
    container (assembling a dataset, scoring a whole run) and one stamped
    frame_id would attribute all of them to a single keyframe.

    `evidence` is the reverse -- it OVERRIDES data["evidence"] when non-empty,
    because that is where a caller records what it just did ("kept by ByteTrack
    id 7"), while the default preserves the detector's original prose.
    """
    if frame_w <= 0 or frame_h <= 0:
        raise SvBridgeError(f"frame size must be positive, got {frame_w}x{frame_h}")
    if d.is_empty():
        return []

    n = len(d)
    keys = class_order(domain)
    sx = BOX_SCALE / frame_w
    sy = BOX_SCALE / frame_h

    # Every column is optional on an sv.Detections built by third-party code (a
    # tracker, a slicer, a from_* loader), so each is resolved independently
    # rather than assuming to_sv produced this one.
    cls_col = _column(d, DATA_CLS, n)
    sev_col = _column(d, DATA_SEVERITY, n)
    fid_col = _column(d, DATA_FRAME_ID, n)
    ev_col = _column(d, DATA_EVIDENCE, n)

    out: list[Detection] = []
    for i in range(n):
        if cls_col is not None:
            cls = str(cls_col[i])
        elif d.class_id is not None:
            idx = int(d.class_id[i])
            if not 0 <= idx < len(keys):
                raise SvBridgeError(
                    f"class_id {idx} is outside domain {domain.key!r} "
                    f"(0..{len(keys) - 1}). It was probably assigned against a "
                    "different taxonomy -- see class_order()."
                )
            cls = keys[idx]
        else:
            raise SvBridgeError(
                "cannot recover cls: this sv.Detections has neither "
                f"data[{DATA_CLS!r}] nor class_id."
            )

        x1, y1, x2, y2 = (float(v) for v in d.xyxy[i])

        severity = DEFAULT_SEVERITY if sev_col is None else int(sev_col[i])
        # Pydantic would reject an out-of-band severity and take the whole batch
        # down with it. Clamping is right for an ORDINAL grade -- saturating at
        # the ends is the honest reading of "off the scale" -- and a tracker or a
        # hand-built container has no reason to respect our 1..5 range.
        severity = max(1, min(5, severity))

        conf = 0.0 if d.confidence is None else float(d.confidence[i])
        # float32 storage can land a hair outside [0, 1] and the schema bound is
        # strict, so clamp rather than fail validation over 1e-7.
        conf = max(0.0, min(1.0, conf))

        out.append(
            Detection(
                frame_id=frame_id if fid_col is None else str(fid_col[i]),
                cls=cls,
                box=[x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                severity=severity,
                confidence=conf,
                evidence=evidence or ("" if ev_col is None else str(ev_col[i])),
                # lat/lon/range_m/anchor stay at their defaults on purpose --
                # see "WHAT IS DELIBERATELY NOT CARRIED ACROSS" in the module
                # docstring.
            )
        )
    return out


def _column(d: sv.Detections, key: str, n: int) -> list[Any] | np.ndarray | None:
    """One data column if present AND of the right length, else None.

    The length check exists because supervision does not police data columns
    added by third-party code, and a short column would silently make this read
    another detection's class or severity -- a mix-up whose output looks
    completely valid.
    """
    col = d.data.get(key)
    if col is None:
        return None
    if len(col) != n:
        raise SvBridgeError(
            f"data[{key!r}] holds {len(col)} entries for {n} detections; "
            "the container was built inconsistently."
        )
    return col
