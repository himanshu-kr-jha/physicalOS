"""Infer defects that consist of something NOT being there.

THE PROBLEM WITH DETECTING ABSENCE
Every other class here is a thing you can point at. "No street lighting" is not.
Asking a per-frame detector "is there a lighting gap?" asks it to reason about
what lies outside the frame, and models answer such questions by guessing --
measured earlier in this project, prompting for absence classes produced 12
hazards where 1 existed.

Absence is a property of a STRETCH OF ROUTE, not of a frame. If no streetlight
was detected anywhere along 66 m, that silence is the evidence, and it is
computable rather than guessable.

HOW IT WORKS
For each rule (see config.AbsenceSpec):
  1. Walk the route accumulating distance per keyframe.
  2. Place every detected instance of the asset class at its route distance.
  3. The gaps are: start -> first asset, between consecutive assets, and last
     asset -> end. A route with no assets at all is one gap of the whole length.
  4. Emit a finding for every gap at least `min_gap_m` long, positioned at the
     gap midpoint, geometry "segment".

WHAT THIS CANNOT KNOW
It infers absence from non-detection, so a missed detection looks identical to a
real gap. Two consequences worth stating plainly:

  - A short route cannot support a confident claim. A 40 m clip with no
    streetlight may simply be the span between two poles, which is normal
    spacing. Confidence therefore scales with how far past the threshold the gap
    runs, and is capped well below certainty.
  - If the backend cannot see the asset at all -- the ONNX model has no
    streetlight class -- then EVERY route looks unlit. Callers pass `detectable`
    so that "we never looked" is never reported as "it is not there".
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import AbsenceSpec, DomainConfig
from .geo import haversine_m
from .schema import Detection, Finding, Frame


@dataclass(frozen=True)
class Gap:
    """One stretch of route where a required asset was never detected."""

    rule: AbsenceSpec
    start_m: float
    end_m: float
    mid_lat: float
    mid_lon: float
    frame_index: int
    # Was the asset seen anywhere at all on this route? Distinguishes "a 40 m
    # unlit stretch" from "the whole route is unlit", which read very differently.
    any_asset: bool

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


def route_distances(frames: list[Frame]) -> list[float]:
    """Cumulative along-route distance in metres, one per keyframe."""
    out = [0.0]
    for i in range(1, len(frames)):
        out.append(
            out[-1]
            + haversine_m(
                frames[i - 1].lat, frames[i - 1].lon, frames[i].lat, frames[i].lon
            )
        )
    return out


def _nearest_frame(frames: list[Frame], lat: float, lon: float) -> int:
    return min(
        range(len(frames)),
        key=lambda i: haversine_m(frames[i].lat, frames[i].lon, lat, lon),
    )


def _at_distance(
    frames: list[Frame], dists: list[float], target: float
) -> tuple[float, float, int]:
    """Interpolate lat/lon at a route distance. Returns (lat, lon, frame_index)."""
    if not frames:
        return 0.0, 0.0, 0
    if target <= 0:
        return frames[0].lat, frames[0].lon, 0
    if target >= dists[-1]:
        return frames[-1].lat, frames[-1].lon, len(frames) - 1

    hi = next(i for i, d in enumerate(dists) if d >= target)
    lo = max(0, hi - 1)
    span = dists[hi] - dists[lo]
    f = 0.0 if span <= 0 else (target - dists[lo]) / span
    return (
        frames[lo].lat + (frames[hi].lat - frames[lo].lat) * f,
        frames[lo].lon + (frames[hi].lon - frames[lo].lon) * f,
        lo if f < 0.5 else hi,
    )


def absence_gaps(
    frames: list[Frame],
    findings: list[Finding],
    domain: DomainConfig,
    detectable: set[str] | None = None,
) -> list[Gap]:
    """Every stretch missing its asset, in route order per rule.

    Split out from infer_absences because the gap's EXTENT is useful on its own.
    The KMZ export draws a line along it -- a single pin would claim a point
    location that does not exist -- and it must be the same stretch the finding
    was derived from, not a second computation that can drift out of step.

    `any_asset` records whether the asset was seen ANYWHERE on the route, which
    is what distinguishes "a 40 m unlit stretch" from "the entire route is unlit".
    """
    if len(frames) < 2 or not domain.absences:
        return []

    dists = route_distances(frames)
    total = dists[-1]
    out: list[Gap] = []

    for rule in domain.absences:
        if detectable is not None and rule.asset not in detectable:
            continue

        # Route distance of every detected instance of the asset.
        at = sorted(
            dists[_nearest_frame(frames, f.lat, f.lon)]
            for f in findings
            if f.cls == rule.asset and f.lat is not None and f.lon is not None
        )

        # Gaps: start->first, between, last->end. No assets at all = one gap.
        gaps: list[tuple[float, float]] = []
        if not at:
            gaps.append((0.0, total))
        else:
            if at[0] >= rule.min_gap_m:
                gaps.append((0.0, at[0]))
            for a, b in zip(at, at[1:]):
                if b - a >= rule.min_gap_m:
                    gaps.append((a, b))
            if total - at[-1] >= rule.min_gap_m:
                gaps.append((at[-1], total))

        for start, end in gaps:
            if end - start < rule.min_gap_m:
                continue
            lat, lon, fidx = _at_distance(frames, dists, (start + end) / 2.0)
            out.append(
                Gap(
                    rule=rule,
                    start_m=start,
                    end_m=end,
                    mid_lat=lat,
                    mid_lon=lon,
                    frame_index=fidx,
                    any_asset=bool(at),
                )
            )

    return out


def gap_polyline(
    frames: list[Frame], gap: Gap, steps: int = 12
) -> list[tuple[float, float]]:
    """Sample a gap's extent as lat/lon points, for drawing it as a line."""
    if len(frames) < 2:
        return []
    dists = route_distances(frames)
    pts: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = gap.start_m + (gap.end_m - gap.start_m) * i / steps
        lat, lon, _ = _at_distance(frames, dists, t)
        pts.append((lat, lon))
    return pts


def infer_absences(
    frames: list[Frame],
    findings: list[Finding],
    domain: DomainConfig,
    detectable: set[str] | None = None,
) -> list[Finding]:
    """Emit a Finding for every stretch missing a required asset.

    `detectable` is the set of asset classes the perception backend could
    plausibly have found. Rules whose asset is not in it are skipped.
    """
    out: list[Finding] = []

    for counter, gap in enumerate(
        absence_gaps(frames, findings, domain, detectable)
    ):
        rule = gap.rule
        length = gap.end_m - gap.start_m
        lat, lon = gap.mid_lat, gap.mid_lon
        frame = frames[gap.frame_index]

        # Confidence rises with how far past the threshold the gap runs, but
        # is capped: absence inferred from non-detection is never certain.
        ratio = length / max(rule.min_gap_m, 1e-6)
        confidence = round(min(0.90, 0.45 + 0.15 * (ratio - 1.0)), 3)

        covered = f"{length:.0f} m" if gap.any_asset else "the entire surveyed route"
        note = f" {rule.note}" if rule.note else ""
        text = (
            f"No {domain.spec(rule.asset).label.lower()} was detected over "
            f"{covered} (threshold {rule.min_gap_m:.0f} m). Inferred from "
            f"absence of detections, not observed directly.{note}"
        )

        # The evidence frame sits at the middle of the gap. Its box spans the
        # whole frame because absence is not localised to a spot -- the
        # viewer renders these without a box for exactly that reason.
        det = Detection(
            frame_id=frame.frame_id,
            cls=rule.key,
            box=[0.0, 0.0, 1000.0, 1000.0],
            severity=rule.severity,
            confidence=confidence,
            evidence=text,
            lat=lat,
            lon=lon,
            range_m=None,
        )

        out.append(
            Finding(
                finding_id=f"a-{counter:04d}",
                cls=rule.key,
                label=rule.label,
                geometry="segment",
                lat=lat,
                lon=lon,
                severity=rule.severity,
                confidence=confidence,
                t_sec=frame.t_sec,
                evidence=[det],
            )
        )

    return out


def coverage_report(
    frames: list[Frame], findings: list[Finding], domain: DomainConfig
) -> list[dict]:
    """Per-asset coverage summary, for the viewer's coverage panel."""
    if len(frames) < 2 or not domain.absences:
        return []

    total = route_distances(frames)[-1]
    rows: list[dict] = []
    for rule in domain.absences:
        found = sum(1 for f in findings if f.cls == rule.asset)
        gaps = [f for f in findings if f.cls == rule.key]
        rows.append(
            {
                "asset": rule.asset,
                "asset_label": domain.spec(rule.asset).label,
                "absence_key": rule.key,
                "absence_label": rule.label,
                "found": found,
                "gaps": len(gaps),
                "route_m": round(total, 1),
                "min_gap_m": rule.min_gap_m,
                "per_km": round(found / (total / 1000.0), 1) if total > 0 else 0.0,
            }
        )
    return rows
