"""Data contracts for the whole system.

This module is the single source of truth. The viewer's types.ts is generated
from here (`pos export-types`), so the pipeline and the UI cannot disagree.

BOX CONVENTION -- pinned here and nowhere else:
    box = [x1, y1, x2, y2]
    normalised to 0..1000, origin TOP-LEFT, x rightward, y downward.
Every detector backend must convert into this before returning.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BOX_SCALE = 1000.0
"""Boxes are normalised to this range on both axes, origin top-left."""

GeometryKind = Literal["point", "segment", "area"]


class Frame(BaseModel):
    """One sampled keyframe, tied to a place on Earth."""

    frame_id: str
    t_sec: float
    ts: str | None = None
    lat: float
    lon: float
    heading_deg: float
    path: str
    width: int
    height: int


class Detection(BaseModel):
    """A single raw observation in a single frame, straight from a detector."""

    frame_id: str
    cls: str
    box: list[float] = Field(min_length=4, max_length=4)
    severity: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""
    # filled in by `localize`; None when the detection is above the horizon
    lat: float | None = None
    lon: float | None = None
    range_m: float | None = None
    # Which pixel row was projected: "bottom" for point defects, "centre" for
    # area/segment classes, whose box bottom is often just where the frame cut
    # them off. Recorded because it changes the resulting range materially.
    anchor: str = "bottom"


class Finding(BaseModel):
    """A deduplicated real-world thing, backed by every frame that saw it."""

    finding_id: str
    cls: str
    label: str
    geometry: GeometryKind = "point"
    lat: float | None = None
    lon: float | None = None
    severity: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    t_sec: float = 0.0
    evidence: list[Detection] = Field(default_factory=list)

    # --- how well do we actually know where this is? -----------------------
    # A blanket "+/-2-4 m" is all a single ground-plane projection can claim. A
    # finding seen from two or more positions can do better, and can say by how
    # much, so these fields let the UI and the report state a real number instead
    # of a disclaimer.
    #
    #   ground_plane  one sighting, flat-ground assumption, pitch-sensitive
    #   triangulated  two or more bearing rays intersected; residual is
    #                 a genuine geometric uncertainty in metres
    #   camera        above the horizon, pinned to the camera, never ranged
    pos_method: str = "ground_plane"
    pos_residual_m: float | None = None
    n_rays: int = 0
    parallax_deg: float | None = None


class Segment(BaseModel):
    """A stretch of route with a quality score. Drives the heatmap."""

    seg_id: int
    start: list[float] = Field(min_length=2, max_length=2)  # [lat, lon]
    end: list[float] = Field(min_length=2, max_length=2)
    length_m: float
    quality_index: float = Field(ge=0.0, le=100.0)
    color: str
    finding_ids: list[str] = Field(default_factory=list)
    t_start: float = 0.0
    t_end: float = 0.0


class Building(BaseModel):
    """OSM building footprint, extruded by the viewer."""

    osm_id: int
    height_m: float
    footprint: list[list[float]]  # [[lat, lon], ...]
    tags: dict[str, str] = Field(default_factory=dict)


class RoadWay(BaseModel):
    """OSM road centreline, for context geometry."""

    osm_id: int
    highway: str
    name: str = ""
    path: list[list[float]]  # [[lat, lon], ...]


class Twin(BaseModel):
    buildings: list[Building] = Field(default_factory=list)
    roads: list[RoadWay] = Field(default_factory=list)


class ScoreSummary(BaseModel):
    quality_index: float
    grade: str
    counts: dict[str, int] = Field(default_factory=dict)
    severity_histogram: dict[str, int] = Field(default_factory=dict)
    total_findings: int = 0
    route_length_m: float = 0.0


class RunManifest(BaseModel):
    """Everything the viewer needs to know about a run, in one file."""

    run_id: str
    domain: str
    domain_label: str
    created: str
    video: str
    origin: list[float] = Field(min_length=2, max_length=2)  # [lat, lon]
    n_frames: int = 0
    n_detections: int = 0
    n_findings: int = 0
    duration_sec: float = 0.0
    backend: str = "mock"
    has_pointcloud: bool = False
    has_twin: bool = False
    summary: ScoreSummary | None = None
