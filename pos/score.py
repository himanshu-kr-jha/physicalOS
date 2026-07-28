"""Turn findings into a per-segment quality score and a route summary.

The score is deliberately simple and fully explainable: every finding applies
a penalty of (class weight x severity x confidence), penalties in a segment are
summed and normalised by length, and the segment score is 100 minus that.
Weights live in the domain YAML, so an engineer can argue with the numbers and
change them without touching code.

There is no learned model here on purpose. An opaque score nobody can
interrogate is worse than no score.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import DomainConfig
from .geo import haversine_m
from .schema import Finding, Frame, ScoreSummary, Segment

DEFAULT_SEGMENT_M = 20.0

# Worst -> best. Each entry is (minimum index, colour).
_RAMP = [
    (0, "#b91c1c"),
    (25, "#dc2626"),
    (40, "#ea580c"),
    (55, "#f59e0b"),
    (70, "#eab308"),
    (82, "#84cc16"),
    (92, "#22c55e"),
]


def color_for(index: float) -> str:
    chosen = _RAMP[0][1]
    for threshold, color in _RAMP:
        if index >= threshold:
            chosen = color
    return chosen


def grade_for(index: float) -> str:
    for cutoff, grade in ((90, "A"), (75, "B"), (60, "C"), (45, "D"), (30, "E")):
        if index >= cutoff:
            return grade
    return "F"


def _penalty(finding: Finding, domain: DomainConfig) -> float:
    """Weight x severity, scaled by confidence so hesitant calls hurt less."""
    weight = domain.spec(finding.cls).weight
    if weight <= 0:
        return 0.0  # assets (streetlight present, compliant worker) never penalise
    return weight * finding.severity * max(0.35, finding.confidence)


def build_segments(
    frames: list[Frame],
    findings: list[Finding],
    domain: DomainConfig,
    segment_m: float = DEFAULT_SEGMENT_M,
) -> list[Segment]:
    """Split the route into fixed-length segments and score each one."""
    if len(frames) < 2:
        return []

    # Walk the route accumulating distance, cutting a boundary every segment_m.
    boundaries: list[int] = [0]
    travelled = 0.0
    for i in range(1, len(frames)):
        travelled += haversine_m(
            frames[i - 1].lat, frames[i - 1].lon, frames[i].lat, frames[i].lon
        )
        if travelled >= segment_m:
            boundaries.append(i)
            travelled = 0.0
    if boundaries[-1] != len(frames) - 1:
        boundaries.append(len(frames) - 1)

    if len(boundaries) < 2:
        return []

    last = len(boundaries) - 2  # index of the final (a, b) pair
    segments: list[Segment] = []

    for s, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        fa, fb = frames[a], frames[b]
        seg_len = sum(
            haversine_m(
                frames[i].lat, frames[i].lon, frames[i + 1].lat, frames[i + 1].lon
            )
            for i in range(a, b)
        )
        t_start, t_end = fa.t_sec, fb.t_sec

        # A finding belongs to the segment whose time window contains its first
        # sighting. Time is a more reliable key than distance, because the
        # finding's projected position is noisier than the route itself. The
        # final segment takes everything at or after its start so nothing is
        # dropped at the end of the route.
        members = [
            f
            for f in findings
            if (t_start <= f.t_sec < t_end) or (s == last and f.t_sec >= t_start)
        ]

        penalty = sum(_penalty(f, domain) for f in members)
        # Normalise by length so a long segment is not unfairly punished.
        density = penalty / max(seg_len / DEFAULT_SEGMENT_M, 0.5)
        # Floor at 5 rather than 0: a saturated 0.0 is indistinguishable from
        # "no data" when you glance at a heatmap, and every stretch we scored
        # did have data.
        index = max(5.0, min(100.0, 100.0 - density * 2.2))

        segments.append(
            Segment(
                seg_id=s,
                start=[fa.lat, fa.lon],
                end=[fb.lat, fb.lon],
                length_m=round(seg_len, 2),
                quality_index=round(index, 1),
                color=color_for(index),
                finding_ids=[f.finding_id for f in members],
                t_start=t_start,
                t_end=t_end,
            )
        )
    return segments


def summarize(
    findings: list[Finding],
    segments: list[Segment],
    domain: DomainConfig,
) -> ScoreSummary:
    counts: dict[str, int] = {}
    hist: dict[str, int] = {}
    for f in findings:
        counts[f.cls] = counts.get(f.cls, 0) + 1
        hist[str(f.severity)] = hist.get(str(f.severity), 0) + 1

    route_len = sum(s.length_m for s in segments)
    if not segments:
        overall = 100.0
    elif route_len > 0:
        # Length-weighted, so a short bad stretch cannot dominate a long road.
        overall = sum(s.quality_index * s.length_m for s in segments) / route_len
    else:
        overall = sum(s.quality_index for s in segments) / len(segments)

    return ScoreSummary(
        quality_index=round(overall, 1),
        grade=grade_for(overall),
        counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        severity_histogram=dict(sorted(hist.items())),
        total_findings=len(findings),
        route_length_m=round(route_len, 1),
    )


def write_segments(segments: list[Segment], path: Path) -> None:
    Path(path).write_text(json.dumps([s.model_dump() for s in segments], indent=2))


def read_segments(path: Path) -> list[Segment]:
    return [Segment(**d) for d in json.loads(Path(path).read_text())]
