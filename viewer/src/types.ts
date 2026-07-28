/**
 * Mirrors pos/schema.py BY HAND. There is no codegen step.
 * If you change a model in pos/schema.py, change it here too.
 *
 * The server ships the domain taxonomy inside /api/manifest (see ClassSpec)
 * specifically so that class labels, colours and alert flags are NOT duplicated
 * in this file -- they come from the YAML at runtime.
 */

/** Boxes are [x1,y1,x2,y2] normalised 0..1000, origin TOP-LEFT. */
export const BOX_SCALE = 1000;

export type GeometryKind = "point" | "segment" | "area";

export interface Frame {
  frame_id: string;
  t_sec: number;
  ts: string | null; // ISO-8601 UTC, e.g. "2026-07-25T09:14:31Z"
  lat: number;
  lon: number;
  heading_deg: number;
  path: string;
  width: number;
  height: number;
}

export interface Detection {
  frame_id: string;
  cls: string;
  box: [number, number, number, number];
  severity: number;
  confidence: number;
  evidence: string;
  lat: number | null;
  lon: number | null;
  range_m: number | null;
  /** "bottom" for point defects, "centre" for area/segment classes. */
  anchor?: string;
}

export interface Finding {
  finding_id: string;
  cls: string;
  label: string;
  geometry: GeometryKind;
  lat: number | null;
  lon: number | null;
  severity: number;
  confidence: number;
  t_sec: number;
  evidence: Detection[];

  /**
   * How well the position is actually known. Optional because runs made before
   * these fields existed are still valid.
   *
   *   ground_plane  one sighting, flat-ground assumption, pitch-sensitive
   *   triangulated  bearing rays from 2+ positions; residual is a real metre figure
   *   camera        above the horizon, pinned to the camera, never ranged
   */
  pos_method?: string;
  pos_residual_m?: number | null;
  n_rays?: number;
  parallax_deg?: number | null;
}

export interface Segment {
  seg_id: number;
  start: [number, number]; // [lat, lon]
  end: [number, number];
  length_m: number;
  quality_index: number;
  color: string;
  finding_ids: string[];
  t_start: number;
  t_end: number;
}

export interface BuildingGeom {
  osm_id: number;
  height_m: number;
  footprint: [number, number][]; // [[lat, lon], ...]
  tags: Record<string, string>;
}

export interface RoadWay {
  osm_id: number;
  highway: string;
  name: string;
  path: [number, number][];
}

export interface Twin {
  buildings: BuildingGeom[];
  roads: RoadWay[];
}

export interface ScoreSummary {
  quality_index: number;
  grade: string;
  counts: Record<string, number>;
  severity_histogram: Record<string, number>;
  total_findings: number;
  route_length_m: number;
}

/** One class from the domain YAML, forwarded by /api/manifest. */
export interface ClassSpec {
  key: string;
  label: string;
  color: string;
  geometry: GeometryKind;
  alert: boolean;
  weight: number;
  /** True when the class is INFERRED from missing coverage, not detected. */
  absence?: boolean;
}

/** One row of /api/coverage: how well an asset is covered along the route. */
export interface CoverageRow {
  asset: string;
  asset_label: string;
  absence_key: string;
  absence_label: string;
  found: number;
  gaps: number;
  route_m: number;
  min_gap_m: number;
  per_km: number;
}

export interface Manifest {
  run_id: string;
  domain: string;
  domain_label: string;
  created: string; // ISO-8601 UTC
  video: string;
  origin: [number, number]; // [lat, lon]
  n_frames: number;
  n_detections: number;
  n_findings: number;
  duration_sec: number;
  backend: string;
  has_pointcloud: boolean;
  /** Whether the source clip is reachable; set by the server. */
  has_video?: boolean;
  has_twin: boolean;
  summary: ScoreSummary | null;
  classes: ClassSpec[];
  index_name: string;
}

/**
 * Cached satellite texture for the ground plane, written by `pos basemap`.
 *
 * `local` is the extent in the same metres-from-origin frame as toLocal(), so
 * the viewer positions one plane and never does tile maths. `attribution` is a
 * licence obligation, not decoration -- it has to stay on screen.
 */
export interface Basemap {
  available: boolean;
  provider?: string;
  attribution?: string;
  /** "satellite" | "street" -- a street map is for checking alignment, not looks. */
  kind?: string;
  zoom?: number;
  size?: { w: number; h: number };
  bbox?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  local?: {
    x_min: number;
    x_max: number;
    z_min: number;
    z_max: number;
    width_m: number;
    height_m: number;
  };
  m_per_px?: number;
  fetched_at?: string; // ISO-8601 UTC
}

/** Payload of a `data:` line on /stream. */
export interface StreamEvent {
  type: "finding" | "done";
  t_sec: number;
  finding?: {
    finding_id: string;
    cls: string;
    label: string;
    lat: number | null;
    lon: number | null;
    severity: number;
    confidence: number;
    geometry: GeometryKind;
    alert: boolean;
    n_evidence: number;
  };
}
