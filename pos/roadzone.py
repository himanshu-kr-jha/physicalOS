"""Two measurements that only the road mask can make: is it ON the road, and how
wide is the road.

WHY THIS EXISTS
pos/segment.py produces a per-frame carriageway mask and pos/svbridge.py converts
our boxes into supervision's pixel space. Neither one draws a conclusion. This
module draws the two conclusions that are worth money to a road authority:

1. GATING. Pavement distress is defined by the surface it sits on. A "pothole"
   is a hole in a carriageway; the same dark bowl-shaped shape in a roadside
   spoil heap, a drain or a shop forecourt is not a pothole, it is a false
   positive -- and today it survives all the way onto the map, because a box
   alone cannot tell those apart. The mask can.
2. WIDTH IN METRES. "The road narrows here" is a VLM opinion. "The carriageway
   is 4.1 m wide at 15 m ahead, down from 6.8 m" is a measurement an engineer
   can act on, and it is the difference between an encroachment report that gets
   filed and one that gets argued with.

WHAT THIS MODULE DOES NOT DO
It does not drop detections and it does not write files. Gating returns a verdict
per detection and the caller decides policy, because "off the carriageway" means
delete for a pothole, means nothing for a streetlight, and means "this is the
interesting case" for a footpath obstruction. It has no CLI wiring and no I/O of
its own: masks come in from pos/segment.py, boxes come in from the caller.

THE TWO INPUTS ARE NOT INTERCHANGEABLE, AND THAT IS DELIBERATE
Gating takes the POLYGON, because sv.PolygonZone is what does point-in-zone
properly (and because a polygon is what the viewer can draw). Width takes the
MASK, because pos/segment.py measured the polygon to over-cover the mask at IoU
0.92-0.95 -- a straight simplified edge cuts the corner off a concave kerb line
and takes a little verge with it. A few pixels of slack is harmless for "is this
point inside", and is a direct error in metres for "how wide is this". Its
docstring says so explicitly; this module obeys it.

ONE CONSEQUENCE OF THAT SLACK, WORTH KNOWING BEFORE TRUSTING A VERDICT
The gate is PERMISSIVE. The zone is slightly larger than the road, so a
rejection is a strong claim (the anchor is outside even the generous outline)
and an acceptance is a weak one. Verdicts therefore also carry on_mask, the
same point tested against the raw mask, so the disagreement band is visible in
the output instead of being a paragraph in a docstring.

AN EMPTY MASK IS A NORMAL OUTCOME
Measured on runs/POC-1: 10-15% road coverage on open carriageway and 0.0% on
frame 00400, a crowded market scene. Every function here treats "no road in this
frame" as a first-class answer -- on_carriageway is None, not False. Nothing was
observed about that detection, and False would launder ignorance into evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median

import numpy as np
import supervision as sv

# cv2 arrives as a hard dependency of supervision, which pyproject already
# requires; there is no separate opencv pin to keep in step. Same reasoning as
# pos/segment.py, which is where MIN_AREA_FRAC is measured and justified -- it is
# imported rather than redefined so the two cannot drift apart.
import cv2

from .config import CameraConfig, DomainConfig
from .geo import focal_px, horizon_v, project_to_ground
from .schema import Detection
from .segment import MIN_AREA_FRAC
from .svbridge import to_sv


class RoadZoneError(ValueError):
    """Raised when a measurement would have to guess to return a number."""


# ---------------------------------------------------------------------------
# WHICH CLASSES ARE CARRIAGEWAY-ONLY
# ---------------------------------------------------------------------------

# The field a domain YAML may set on a class to state its zone outright. Nothing
# in configs/domains/*.yaml sets it today, and pos/config.py ClassSpec has no such
# field, so reading it is forward compatibility, not current behaviour: adding
#     zone: carriageway
# to a ClassSpec would make the derivation below unnecessary for that class, and
# this module already prefers it. That is the durable fix and it belongs in
# pos/config.py, which this task does not own.
ZONE_FIELD = "zone"
ZONE_CARRIAGEWAY = "carriageway"

# Vocabulary, NOT a class list. The brief is to read the domain rather than
# hardcode the classes, and the honest description of what follows is: the
# CLASSES are read from the domain (key, label and the `hint` prose the domain
# author wrote for the VLM), and the TERMS are hardcoded. That is weaker than a
# declared zone field and stronger than a list of class keys, because it keeps
# working when a domain YAML gains a class -- a new "spalling" or "manhole_sunk"
# classifies itself from its own hint, whereas a hardcoded key list would silently
# stop gating it.
#
# Terms are matched at a word start (\bcrack also matches "cracking", "cracks").
CARRIAGEWAY_TERMS: tuple[str, ...] = (
    "carriageway",
    "crack",
    "pothole",
    "rut",
    "wheel path",
    "wheel track",
    "asphalt",
    "bitumen",
    "tarmac",
    "chipseal",
    "aggregate",
    "surface",
)

# Checked FIRST and wins outright, because several genuinely off-carriageway
# classes describe themselves by reference to the carriageway they run beside:
# road_pci's footpath_damaged hint is "paving alongside the carriageway", and
# footpath_missing's note is "pedestrians are forced onto the carriageway".
# Without this precedence rule both would gate as carriageway-only and every real
# footpath defect would be discarded as a false positive -- the exact opposite of
# the intended effect.
NON_CARRIAGEWAY_TERMS: tuple[str, ...] = (
    "footpath",
    "footway",
    "sidewalk",
    "walkway",
    "pedestrian",
    "verge",
    "kerb",
    "curb",
    "roadside",
    "median",
    "central reserve",
)

# "pavement" is deliberately in NEITHER list. It is the one word here that means
# opposite things in the two dialects this codebase mixes: the road structure in
# the PCI / ASTM D6433 sense that configs/domains/road_pci.yaml uses ("pavement
# distress", "pavement edge"), and the footpath in ordinary British usage. A term
# that inverts meaning between two readers of the same YAML cannot be evidence
# either way, so it is ignored and the surrounding words decide.


@dataclass(frozen=True)
class ZoneRule:
    """Why one class was, or was not, treated as carriageway-only.

    The reason is carried rather than discarded because this is a heuristic over
    prose. When it gets a class wrong the first question is always "on what
    grounds", and an audit table of (cls, verdict, source, matched) answers it in
    one glance instead of a re-read of the YAML.
    """

    cls: str
    carriageway_only: bool
    source: str  # override | domain_field | absence | lexical | default
    matched: str = ""


def carriageway_rules(
    domain: DomainConfig,
    overrides: Mapping[str, bool] | None = None,
) -> dict[str, ZoneRule]:
    """Per-class zone rules for every class in the domain, including absences.

    Precedence, highest first:
      1. `overrides` -- an explicit caller decision, for the case where an
         operator disagrees with the derivation and needs it fixed now rather
         than after a config change.
      2. ClassSpec.zone, if a future pos/config.py grows the field (see
         ZONE_FIELD). Declared beats derived, always.
      3. Absence classes are NEVER carriageway-only, on structural grounds
         rather than lexical ones: pos/absence.py:224 emits them with a
         FULL-FRAME box [0, 0, 1000, 1000], so their anchor is a fixed point in
         the middle of the bottom edge of the frame and says nothing about where
         the absence is. Gating them would test the road mask at a pixel chosen
         by the box convention. They are excluded before any prose is read.
      4. The lexical derivation over key + label + hint.
    """
    over = dict(overrides or {})
    # Absence keys are the ones class_map synthesises: present in class_map, not
    # in the authored `classes` list. Deriving the set this way rather than from
    # domain.absences keeps it correct if a YAML ever declares a class of the
    # same key explicitly (class_map's setdefault lets the authored one win).
    authored = {c.key for c in domain.classes}

    rules: dict[str, ZoneRule] = {}
    for key, spec in domain.class_map.items():
        if key in over:
            rules[key] = ZoneRule(key, bool(over[key]), "override")
            continue

        declared = getattr(spec, ZONE_FIELD, None)
        if declared:
            rules[key] = ZoneRule(
                key, str(declared) == ZONE_CARRIAGEWAY, "domain_field", str(declared)
            )
            continue

        if key not in authored:
            rules[key] = ZoneRule(key, False, "absence")
            continue

        # The hint is the domain author's own description of the class, written
        # for the VLM. It is the richest zone evidence the YAML actually holds.
        text = f"{spec.key} {spec.label} {spec.hint}".lower()

        hit = _first_term(text, NON_CARRIAGEWAY_TERMS)
        if hit:
            rules[key] = ZoneRule(key, False, "lexical", hit)
            continue
        hit = _first_term(text, CARRIAGEWAY_TERMS)
        if hit:
            rules[key] = ZoneRule(key, True, "lexical", hit)
            continue

        # Nothing matched: NOT carriageway-only. The default has to be "do not
        # gate", because a wrong True deletes real findings while a wrong False
        # only fails to delete false ones. road_pci's `hazard` lands here and
        # that is right -- an open manhole is as much a hazard on a footpath as
        # in a traffic lane.
        rules[key] = ZoneRule(key, False, "default")
    return rules


def carriageway_classes(
    domain: DomainConfig,
    overrides: Mapping[str, bool] | None = None,
) -> set[str]:
    """The class keys that are meaningless off the drivable surface.

    On configs/domains/road_pci.yaml this returns the 11 PCI distress classes
    plus waterlogging -- 12 of 19 -- and excludes hazard, garbage, streetlight,
    both footpath asset classes, both footpath defect classes and both absence
    classes. That split was checked against the YAML by hand; see
    carriageway_rules() for the per-class grounds.

    This is a property of the TAXONOMY, not of whichever model is loaded: a
    pothole is off-carriageway nonsense whether the local detector or the VLM
    reported it. Swapping post_cons does not change this set.
    """
    return {
        k for k, r in carriageway_rules(domain, overrides).items() if r.carriageway_only
    }


def _first_term(text: str, terms: Iterable[str]) -> str:
    for term in terms:
        # \b at the START only. Suffixes are how these words vary in the hints
        # ("crack" -> "cracking", "rut" -> "rutting"), while a trailing \b would
        # miss every one of them. Prefixes are not: matching "crack" inside
        # another word is the risk this guards against.
        if re.search(r"\b" + re.escape(term), text):
            return term
    return ""


# ---------------------------------------------------------------------------
# (1) ON-CARRIAGEWAY GATING
# ---------------------------------------------------------------------------

# The pixel of a box that is tested against the zone.
#
# BOTTOM_CENTER for a point defect, because that is where the object meets the
# ground -- and, more importantly, because it is the SAME pixel pos/localize
# projects: sv.Position.BOTTOM_CENTER is ((x1+x2)/2, y2), which is exactly
# pos/geo.py box_ground_anchor(geometry="point"). The zone verdict and the
# mapped position are therefore derived from one pixel, so they cannot disagree
# -- a detection can never be placed at a lat/lon computed from the carriageway
# while being judged off it.
#
# CENTER for segment/area geometry, for the same parity reason and not as a
# refinement: box_ground_anchor deliberately switches to the box centre for those
# classes (pos/geo.py:147) because their bottom edge is usually just where the
# frame cut the region off. Testing the bottom edge here while ranging the centre
# there would give two answers about one detection.
ANCHOR_POINT = sv.Position.BOTTOM_CENTER
ANCHOR_SPAN = sv.Position.CENTER


@dataclass(frozen=True)
class ZoneVerdict:
    """One detection's relationship to the carriageway.

    on_carriageway is TRISTATE and the third state is load-bearing:
      True   the anchor is inside the road zone
      False  the anchor is outside it -- for a carriageway-only class, evidence
             of a false positive
      None   no opinion. Either the class is not carriageway-only (nothing to
             check) or no road was found in this frame (nothing to check it
             against). None must never be collapsed to False; doing so turns
             "the segmenter saw no road" into "this pothole is not on the road"
             and quietly deletes real findings on every market-scene frame.
    """

    index: int  # position in the dets list that was passed in
    frame_id: str
    cls: str
    carriageway_only: bool
    on_carriageway: bool | None
    on_mask: bool | None  # same point against the raw mask; None if no mask given
    anchor_px: tuple[float, float]
    anchor: str  # "bottom" | "centre" -- mirrors Detection.anchor
    reason: str

    @property
    def rejected(self) -> bool:
        """True only for a carriageway-only class proven to be off the road."""
        return self.carriageway_only and self.on_carriageway is False


@dataclass(frozen=True)
class FrameGate:
    frame_id: str
    road_found: bool
    verdicts: tuple[ZoneVerdict, ...] = ()

    @property
    def n_checked(self) -> int:
        return sum(1 for v in self.verdicts if v.on_carriageway is not None)

    @property
    def n_rejected(self) -> int:
        return sum(1 for v in self.verdicts if v.rejected)

    def rejected_indices(self) -> list[int]:
        return [v.index for v in self.verdicts if v.rejected]

    def keep_mask(self) -> list[bool]:
        """Per-detection keep flag under the strictest reasonable policy.

        Provided as a convenience, not as the policy: it drops exactly the
        `rejected` rows and keeps every None. A caller wanting the opposite
        (surface off-carriageway distress for review rather than delete) reads
        `verdicts` instead.
        """
        return [not v.rejected for v in self.verdicts]


def gate_detections(
    dets: Sequence[Detection],
    polygon: np.ndarray | None,
    frame_w: int,
    frame_h: int,
    domain: DomainConfig,
    road_mask: np.ndarray | None = None,
    rules: Mapping[str, ZoneRule] | None = None,
    frame_id: str = "",
) -> FrameGate:
    """Decide, per detection, whether it sits on the drivable surface.

    Args:
        dets: this frame's detections, boxes normalised 0..1000 (pos/schema.py).
        polygon: (N, 2) int ABSOLUTE PIXELS from RoadSegmenter.polygon(mask), or
            None when that returned None. None is normal, not an error.
        frame_w, frame_h: the size of the image the boxes were measured on. Read
            them from the frames.json entry for THIS frame_id -- pos/svbridge.py
            documents at length why a wrong pair misplaces every box in silence.
        domain: supplies both the class ids and, via carriageway_rules(), which
            classes are carriageway-only.
        road_mask: optional bool (H, W) from RoadSegmenter.mask(). Only used for
            the on_mask cross-check, never for the verdict, so that the verdict
            is exactly what sv.PolygonZone says.
        rules: precomputed carriageway_rules(domain), to avoid re-deriving them
            per frame over a 570-frame run.
        frame_id: fallback only; each Detection carries its own.

    The polygon is NOT re-derived from the mask here even when both are passed.
    Contour extraction lives in pos/segment.py and a second copy of it would
    drift from the first, which is the failure its own docstring calls out.
    """
    if not dets:
        return FrameGate(frame_id=frame_id, road_found=polygon is not None)

    fid = frame_id or dets[0].frame_id
    rule_map = dict(rules) if rules is not None else carriageway_rules(domain)

    # One conversion, through the bridge, so the 0..1000 -> pixel arithmetic
    # exists in exactly one place in this repo.
    sv_dets = to_sv(list(dets), frame_w, frame_h, domain)

    # Anchor coordinates come from sv itself rather than from a second
    # computation, so that on_carriageway and on_mask are testing the SAME
    # float32-derived pixel. Recomputing them in float64 here would differ from
    # PolygonZone's by ~1e-4 px, which np.ceil can turn into a whole pixel and
    # hence into a spurious polygon-vs-mask disagreement at a road edge.
    anchors = {
        ANCHOR_POINT: sv_dets.get_anchors_coordinates(ANCHOR_POINT),
        ANCHOR_SPAN: sv_dets.get_anchors_coordinates(ANCHOR_SPAN),
    }

    zones: dict[sv.Position, np.ndarray] = {}
    if polygon is not None:
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise RoadZoneError(
                f"polygon must be (N>=3, 2) pixel points, got shape {polygon.shape}"
            )
        for pos in (ANCHOR_POINT, ANCHOR_SPAN):
            # A separate zone per anchor because PolygonZone with several
            # triggering_anchors requires ALL of them inside (np.all over the
            # anchor axis). Here the anchor is chosen per detection by geometry,
            # which is a different question, so it needs a separate trigger.
            zone = sv.PolygonZone(polygon=polygon, triggering_anchors=(pos,))
            zones[pos] = zone.trigger(detections=sv_dets)

    verdicts: list[ZoneVerdict] = []
    for i, det in enumerate(dets):
        spec = domain.spec(det.cls)
        pos, anchor_name = (
            (ANCHOR_POINT, "bottom")
            if spec.geometry == "point"
            else (ANCHOR_SPAN, "centre")
        )
        ax, ay = (float(v) for v in anchors[pos][i])
        rule = rule_map.get(det.cls, ZoneRule(det.cls, False, "default"))

        if not rule.carriageway_only:
            on_zone: bool | None = None
            reason = f"{det.cls} is not carriageway-only ({rule.source})"
        elif polygon is None:
            on_zone = None
            reason = "no road polygon in this frame; nothing to test against"
        else:
            on_zone = bool(zones[pos][i])
            reason = (
                f"anchor {anchor_name} ({ax:.0f},{ay:.0f}) "
                f"{'inside' if on_zone else 'outside'} road zone"
            )

        on_mask: bool | None = None
        if road_mask is not None and road_mask.size:
            # np.ceil to match PolygonZone.trigger exactly, so that any
            # polygon/mask disagreement is about geometry and never rounding.
            mx = int(np.ceil(ax))
            my = int(np.ceil(ay))
            h, w = road_mask.shape[:2]
            on_mask = bool(0 <= mx < w and 0 <= my < h and road_mask[my, mx])
            if on_zone is True and on_mask is False:
                # The permissive band the module docstring warns about: inside
                # the simplified outline, not on a predicted road pixel.
                reason += " (in polygon, not on mask: inside the ~5% slack)"

        verdicts.append(
            ZoneVerdict(
                index=i,
                frame_id=det.frame_id or fid,
                cls=det.cls,
                carriageway_only=rule.carriageway_only,
                on_carriageway=on_zone,
                on_mask=on_mask,
                anchor_px=(ax, ay),
                anchor=anchor_name,
                reason=reason,
            )
        )

    return FrameGate(
        frame_id=fid, road_found=polygon is not None, verdicts=tuple(verdicts)
    )


# ---------------------------------------------------------------------------
# (2) CARRIAGEWAY WIDTH IN METRES
# ---------------------------------------------------------------------------

# Forward ranges at which width is measured, in metres. Not pixel rows: a row
# means nothing across cameras, and these numbers are what an engineer reads.
#
# The lower end is bounded by the camera, not by taste. The visible lateral span
# at range Z is 2*Z*tan(hfov/2), which for the bike_kohima preset used by
# runs/POC-1 (hfov 53.04 deg) is almost exactly Z metres -- so a 7 m carriageway
# cannot fit in the frame nearer than 7 m, and every measurement below that is
# clipped by the frame edge rather than by the kerb. 5.0 is kept in the default
# set on purpose: it demonstrates the clipping flag doing its job instead of
# hiding the effect behind a tuned constant.
DEFAULT_RANGES_M: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0)

# The band a measured carriageway has to fall in to be believed. Deliberately
# wide: a single-track lane can genuinely be ~2.5 m and a divided highway seen
# kerb-to-kerb can be ~30 m, and the job here is to catch calibration nonsense
# (0.04 m, 400 m), not to second-guess an unusual but real road.
MIN_PLAUSIBLE_WIDTH_M = 1.5
MAX_PLAUSIBLE_WIDTH_M = 40.0


@dataclass(frozen=True)
class WidthSample:
    """One cross-section of the carriageway, measured on one pixel row.

    span_m is kerb-to-kerb: the leftmost to the rightmost road pixel on the row.
    free_m is the widest CONTIGUOUS run of road on it. They differ when something
    the segmenter excluded stands in the road -- a parked car, a stall, a heap of
    material -- and that difference (span_m - free_m) is the encroachment,
    measured in metres rather than asserted. span_m is the road; free_m is what
    is left of it.

    ok is the only thing a caller should aggregate on. A sample can carry
    numbers and still be untrustworthy: see `note`.
    """

    range_m: float  # requested
    row: int  # pixel row actually measured, -1 when none exists
    range_actual_m: float | None  # range of that integer row
    span_m: float | None
    free_m: float | None
    left_px: int
    right_px: int
    m_per_px: float
    clipped_left: bool
    clipped_right: bool
    ok: bool
    note: str


@dataclass(frozen=True)
class FrameWidths:
    frame_id: str
    coverage: float  # road fraction of the frame, for context on a thin result
    samples: tuple[WidthSample, ...] = ()

    @property
    def usable(self) -> list[WidthSample]:
        return [s for s in self.samples if s.ok]


@dataclass(frozen=True)
class WidthSummary:
    n_frames: int
    n_samples: int
    n_usable: int
    n_clipped: int
    n_no_road: int
    n_no_row: int  # above the horizon, off the frame, or beyond max_range_m
    median_span_m: float | None
    median_free_m: float | None
    min_span_m: float | None
    min_at_frame: str
    min_at_range_m: float | None
    max_span_m: float | None
    # (frame_id, range_m, span_m) for every sample that fed the aggregates, so a
    # surprising median can be traced to the frames that produced it.
    samples_used: tuple[tuple[str, float, float], ...] = field(default=())
    # Rows that DID find road and produced a number, which was then rejected as
    # physically implausible. A high count here is a statement about the camera
    # calibration, not about the road, and deserves to be read separately from
    # "no road on this row" -- the two have completely different remedies.
    n_implausible: int = 0


def row_for_range(range_m: float, frame_h: int, cam: CameraConfig) -> float:
    """The pixel row whose ground projection is `range_m` ahead.

    The exact algebraic inverse of pos/geo.py project_to_ground, not an
    approximation of it: that function computes forward = h / tan(atan2(v - cy, f))
    which reduces to f*h / (v - cy), so v = cy + f*h / Z. Deriving the row this
    way rather than scanning rows keeps the two directions consistent to the
    float, and keeps the horizon definition (pitch_offset_frac) in one place.
    """
    if range_m <= 0:
        raise RoadZoneError(f"range must be positive, got {range_m}")
    return (
        horizon_v(frame_h, cam)
        + focal_px(frame_h, cam.vfov_deg) * cam.height_m / range_m
    )


def carriageway_width(
    road_mask: np.ndarray,
    cam: CameraConfig,
    frame_id: str = "",
    ranges_m: Sequence[float] = DEFAULT_RANGES_M,
    largest_component: bool = True,
    min_area_frac: float = MIN_AREA_FRAC,
) -> FrameWidths:
    """Measure carriageway width in metres at each of `ranges_m`.

    THE GEOMETRY, AND WHY IT IS EXACT RATHER THAN APPROXIMATE
    For a pinhole camera at height h looking at flat ground, a ground point at
    forward X and lateral Y projects to v = cy + f*h/X and u = cx + f*Y/X. Fix a
    row v and X is fixed with it, so Y = (u - cx) * X / f holds exactly across
    the whole row -- there is no small-angle step here. Width in metres is
    therefore just a pixel count times X/f.

    Focal length comes from vfov (pos/geo.py focal_px) and is used for the
    HORIZONTAL scaling, which is only valid for square pixels. Checked against
    the preset rather than assumed: bike_kohima declares vfov 31.36 and hfov
    53.04 on 1920x1080, and f from vfov implies hfov = 2*atan(960/f) = 53.0 deg.
    A 0.1% disagreement, so the pixels are square and one f serves both axes.

    THE THREE ASSUMPTIONS THAT CAN MAKE A CONFIDENT NUMBER WRONG
    1. FLAT GROUND. A crest or a dip changes the true range of a row, and width
       scales linearly with range, so a 10% range error is a 10% width error.
    2. ZERO ROLL. Neither CameraConfig nor scripts/calibrate_from_motion.py has
       a roll term, so an image row is assumed to be a line of constant ground
       range. With roll it cuts the road diagonally and span_m over-reports by
       1/cos(roll) -- 1.5% at 10 degrees, so this is a small error unless the
       camera is visibly tilted.
    3. MASK RESOLUTION. road_seg_unet predicts on a 256x256 grid, so a mask edge
       snaps to ~4.2 rows of a 1080-tall frame. dZ/dv = -f*h/(v-cy)^2, which at
       20 m on the bike_kohima preset is 0.19 m per row: about +/-0.4 m of range
       and hence ~2% of width. Fine for an encroachment report, useless for a
       boundary dispute.

    THE TWO FAILURE MODES THAT ARE FLAGGED RATHER THAN GUESSED AT
      * NO ROW EXISTS for the requested range: the row is at or above the horizon
        (v - cy <= 0), or beyond cam.max_range_m where pos/geo.py refuses to
        project, or below the bottom of the frame (the range is nearer than the
        bonnet). ok=False, span_m=None. Nothing is extrapolated.
      * THE MASK REACHES A FRAME EDGE on that row. The road continues out of
        shot, so the measurement is a LOWER BOUND, not a width. ok=False with
        clipped_left/clipped_right set and the number still reported, because a
        lower bound is useful as long as it is labelled -- silently publishing it
        as a width is what this flag exists to prevent. Expect this at short
        range on any normal lens; see DEFAULT_RANGES_M.

    Frame dimensions are taken from road_mask.shape, never from an argument.
    pos/segment.py's mask() is documented as (row, col) while frames.json is
    (width, height), and a transposed pair here would silently measure the road's
    height. Reading the shape removes the chance to get it wrong.
    """
    if road_mask is None or road_mask.size == 0:
        raise RoadZoneError("road_mask is empty; cannot measure width")
    if road_mask.ndim != 2:
        raise RoadZoneError(
            f"road_mask must be 2-D (H, W), got shape {road_mask.shape}"
        )

    mask = road_mask.astype(bool)
    frame_h, frame_w = mask.shape
    coverage = float(np.count_nonzero(mask)) / float(mask.size)

    if largest_component:
        mask = _largest_component(mask, min_area_frac)

    f = focal_px(frame_h, cam.vfov_deg)
    cx = frame_w / 2.0

    samples: list[WidthSample] = []
    for want in ranges_m:
        v = row_for_range(want, frame_h, cam)
        row = int(round(v))
        if row < 0 or row >= frame_h:
            samples.append(
                _no_row(want, f"row {v:.0f} is outside the frame (0..{frame_h - 1})")
            )
            continue

        # Round-trip through pos/geo.py rather than trusting the requested range:
        # it applies the horizon test and the max_range_m cut-off, and it is what
        # pos/localize uses, so an out-of-range row is rejected here on exactly
        # the same terms a detection on that row would be.
        ground = project_to_ground(cx, float(row), frame_w, frame_h, cam)
        if ground is None:
            samples.append(
                _no_row(
                    want,
                    f"row {row} has no ground range "
                    f"(above the horizon, or beyond max_range_m={cam.max_range_m} m)",
                    row=row,
                )
            )
            continue
        actual, _lateral = ground

        row_mask = mask[row]
        xs = np.flatnonzero(row_mask)
        m_per_px = actual / f
        if xs.size == 0:
            samples.append(
                WidthSample(
                    range_m=want,
                    row=row,
                    range_actual_m=actual,
                    span_m=None,
                    free_m=None,
                    left_px=-1,
                    right_px=-1,
                    m_per_px=m_per_px,
                    clipped_left=False,
                    clipped_right=False,
                    ok=False,
                    note="no road pixels on this row",
                )
            )
            continue

        left, right = int(xs[0]), int(xs[-1])
        # +1 because pixel indices are cell centres: a run from column 10 to
        # column 19 covers ten cells, not nine. Worth 0.01 m at 20 m on this
        # camera -- immaterial to the answer, and wrong in principle otherwise.
        span_px = right - left + 1
        free_px = max(e - s + 1 for s, e in _runs(row_mask))

        clipped_left = left == 0
        clipped_right = right == frame_w - 1
        note = ""
        if clipped_left or clipped_right:
            sides = " and ".join(
                s for s, on in (("left", clipped_left), ("right", clipped_right)) if on
            )
            note = f"mask reaches the {sides} frame edge: LOWER BOUND, not a width"

        span_m = span_px * m_per_px

        # PLAUSIBILITY, AND WHY IT IS NOT PARANOIA
        # The arithmetic above is exact GIVEN the camera model, which means a
        # wrong calibration comes out as a confident wrong number rather than as
        # an error. Measured on runs/POC-1 with the bike_kohima preset: median
        # carriageway 0.20 m, narrowest 0.04 m, reported without a single
        # warning -- because that preset puts the horizon at row 190 of 1080, so
        # the 10 m row lands near the vanishing point where the road is a
        # 37-pixel sliver. No road this pipeline is pointed at is 20 cm wide,
        # and none is 40 m. A sample outside the band is evidence about the
        # CALIBRATION, not about the road, so it must not reach a headline.
        implausible = not (MIN_PLAUSIBLE_WIDTH_M <= span_m <= MAX_PLAUSIBLE_WIDTH_M)
        if implausible:
            reason = (
                f"span {span_m:.2f} m is outside the plausible "
                f"{MIN_PLAUSIBLE_WIDTH_M}-{MAX_PLAUSIBLE_WIDTH_M} m band: the "
                f"camera calibration probably does not match this footage"
            )
            note = f"{note}; {reason}" if note else reason

        samples.append(
            WidthSample(
                range_m=want,
                row=row,
                range_actual_m=actual,
                span_m=span_m,
                free_m=free_px * m_per_px,
                left_px=left,
                right_px=right,
                m_per_px=m_per_px,
                clipped_left=clipped_left,
                clipped_right=clipped_right,
                ok=not (clipped_left or clipped_right or implausible),
                note=note,
            )
        )

    return FrameWidths(frame_id=frame_id, coverage=coverage, samples=tuple(samples))


def width_summary(frames: Iterable[FrameWidths]) -> WidthSummary:
    """Aggregate per-frame cross-sections into the numbers a report quotes.

    Median, not mean: a single frame where the mask leaked onto a side road or a
    forecourt produces a span several times too large, and a mean carries that
    into the headline number while a median ignores it. The minimum is reported
    with its frame and range because "the narrowest point" is only actionable if
    you can go and look at it.

    Only ok samples are aggregated -- clipped rows are lower bounds and would
    drag the median down while looking like ordinary measurements.
    """
    frames = list(frames)
    n_samples = n_usable = n_clipped = n_no_road = n_no_row = n_implausible = 0
    spans: list[float] = []
    frees: list[float] = []
    used: list[tuple[str, float, float]] = []
    worst: tuple[float, str, float] | None = None

    for fw in frames:
        for s in fw.samples:
            n_samples += 1
            if s.span_m is None:
                if s.row < 0 or s.range_actual_m is None:
                    n_no_row += 1
                else:
                    n_no_road += 1
                continue
            if s.clipped_left or s.clipped_right:
                n_clipped += 1
                continue
            if not s.ok:
                # `ok` is the single source of truth, per WidthSample: a sample
                # can carry numbers and still be untrustworthy. Re-deriving
                # usability from the raw fields here readmitted exactly those --
                # a 0.20 m "carriageway" that the plausibility band had already
                # rejected still reached the median, because this loop never
                # asked. Two rules for one concept is how that happens.
                n_implausible += 1
                continue
            n_usable += 1
            spans.append(s.span_m)
            if s.free_m is not None:
                frees.append(s.free_m)
            used.append((fw.frame_id, s.range_m, s.span_m))
            if worst is None or s.span_m < worst[0]:
                worst = (s.span_m, fw.frame_id, s.range_m)

    return WidthSummary(
        n_frames=len(frames),
        n_samples=n_samples,
        n_usable=n_usable,
        n_clipped=n_clipped,
        n_no_road=n_no_road,
        n_no_row=n_no_row,
        n_implausible=n_implausible,
        median_span_m=median(spans) if spans else None,
        median_free_m=median(frees) if frees else None,
        min_span_m=worst[0] if worst else None,
        min_at_frame=worst[1] if worst else "",
        min_at_range_m=worst[2] if worst else None,
        max_span_m=max(spans) if spans else None,
        samples_used=tuple(used),
    )


def _no_row(want: float, note: str, row: int = -1) -> WidthSample:
    return WidthSample(
        range_m=want,
        row=row,
        range_actual_m=None,
        span_m=None,
        free_m=None,
        left_px=-1,
        right_px=-1,
        m_per_px=0.0,
        clipped_left=False,
        clipped_right=False,
        ok=False,
        note=note,
    )


def _largest_component(mask: np.ndarray, min_area_frac: float) -> np.ndarray:
    """Keep only the biggest road blob, as RoadSegmenter.polygon() does.

    Not tidiness: a speckle of tarmac glimpsed between two people at the far left
    of a row, plus the real carriageway at the right, gives a leftmost-to-
    rightmost span across BOTH and a width several metres too large. Restricting
    to one component makes span_m a cross-section of one road.

    Note this keeps the split-by-vehicle case working: a car standing in the road
    punches a hole in the mask but the road remains one 8-connected component
    around it, so both runs on that row survive and span_m still spans the
    carriageway while free_m reports what is left of it.

    8-connectivity to match cv2.findContours, which treats the foreground as
    8-connected -- so this selects the same component RoadSegmenter.polygon()
    would, and the gate and the width are talking about one road.
    """
    if not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if areas[best - 1] < min_area_frac * mask.size:
        # Below pos/segment.py's measured floor this is speckle, not a road, and
        # polygon() would have returned None for the same frame. Returning an
        # empty mask keeps the two answers consistent.
        return np.zeros_like(mask)
    return labels == best


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) column pairs of each contiguous True run."""
    flags = np.concatenate(([False], row.astype(bool), [False])).astype(np.int8)
    d = np.diff(flags)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))
