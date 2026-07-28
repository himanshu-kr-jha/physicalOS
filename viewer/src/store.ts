import { useMemo } from "react";
import { create } from "zustand";
import type {
  Basemap,
  ClassSpec,
  CoverageRow,
  Finding,
  Frame,
  Manifest,
  Segment,
  StreamEvent,
  Twin,
} from "./types";

/**
 * One store for one run. map3d splits state across areaStore/carStore/
 * exportStore, but every field here belongs to a single inspection run, so
 * splitting would only add indirection.
 */

// --------------------------------------------------------------------------
// Local ENU projection
//
// three.js is Y-up. We map the world so +X is EAST and +Z is SOUTH
// (z = -north), keeping a right-handed frame with Y up. Everything in the scene
// is in METRES, so on-screen distances are real distances.
// --------------------------------------------------------------------------

export const M_PER_DEG_LAT = 111320;

export function mPerDegLon(lat: number): number {
  return M_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180);
}

export type XZ = [number, number];

export function toLocal(lat: number, lon: number, origin: [number, number]): XZ {
  const east = (lon - origin[1]) * mPerDegLon(origin[0]);
  const north = (lat - origin[0]) * M_PER_DEG_LAT;
  return [east, -north];
}

// --------------------------------------------------------------------------
// Where the camera vehicle was at a given moment
//
// Frames are keyframes -- typically one every second or two -- so the playback
// head almost never lands exactly on one. Everything that has to move smoothly
// (the chase camera, the ego marker) interpolates between the pair straddling
// `t` rather than snapping to the nearest keyframe, otherwise the map lurches
// once per keyframe.
// --------------------------------------------------------------------------

/** Index of the last frame at or before `t`; -1 when `t` precedes the run. */
export function frameIndexAt(frames: Frame[], t: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t_sec <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

export interface Pose {
  lat: number;
  lon: number;
  /** Degrees clockwise from north, taken from the frame we are travelling FROM. */
  heading_deg: number;
  /** Index of that frame, for slicing the covered part of the route. */
  index: number;
}

/**
 * Interpolated position along the route at video time `t`.
 *
 * Heading is not interpolated. It wraps at 360 degrees, so lerping it sends the
 * marker spinning the long way round whenever the track crosses north; the
 * keyframe's own heading is close enough for a marker a couple of metres wide.
 */
export function poseAt(frames: Frame[], t: number): Pose | null {
  if (frames.length === 0) return null;

  const i = frameIndexAt(frames, t);
  if (i < 0) {
    const f = frames[0];
    return { lat: f.lat, lon: f.lon, heading_deg: f.heading_deg, index: 0 };
  }
  const a = frames[i];
  const b = frames[i + 1];
  if (!b) return { lat: a.lat, lon: a.lon, heading_deg: a.heading_deg, index: i };

  const span = b.t_sec - a.t_sec;
  const u = span > 0 ? Math.min(Math.max((t - a.t_sec) / span, 0), 1) : 0;
  return {
    lat: a.lat + (b.lat - a.lat) * u,
    lon: a.lon + (b.lon - a.lon) * u,
    heading_deg: a.heading_deg,
    index: i,
  };
}

// --------------------------------------------------------------------------

export type LayerKey =
  | "satellite"
  | "cloud"
  | "buildings"
  | "heatmap"
  | "markers"
  | "route";

/**
 * overlay -- video floats over the 3D view, map gets the whole window.
 * split   -- map and video sit side by side, neither hiding the other.
 */
export type Layout = "overlay" | "split";

export interface State {
  loading: boolean;
  error: string | null;

  manifest: Manifest | null;
  /** Satellite ground texture, if `pos basemap` has been run. */
  basemap: Basemap | null;
  findings: Finding[];
  segments: Segment[];
  frames: Frame[];
  twin: Twin;
  coverage: CoverageRow[];

  /** Class key -> spec, built from manifest.classes. */
  classMap: Record<string, ClassSpec>;

  selected: Finding | null;
  hovered: string | null;
  classFilter: Set<string>;
  layers: Record<LayerKey, boolean>;
  layout: Layout;

  /** Playback head, in seconds of video time. */
  playT: number;
  playing: boolean;
  /**
   * Chase camera: while the head is moving, pan the view along the route so the
   * stretch being inspected stays under the camera. Off means the view is
   * wherever the user left it, which is what you want when studying one spot.
   */
  follow: boolean;
  /** Findings revealed by the live stream, by id. */
  streamed: Set<string>;
  streaming: boolean;
  alerts: { id: string; label: string; severity: number; t: number }[];

  loadRun: () => Promise<void>;
  select: (f: Finding | null) => void;
  hover: (id: string | null) => void;
  toggleClass: (key: string) => void;
  clearFilter: () => void;
  toggleLayer: (key: LayerKey) => void;
  setLayout: (l: Layout) => void;
  setPlayT: (t: number) => void;
  setPlaying: (p: boolean) => void;
  toggleFollow: () => void;
  startStream: (speed?: number) => void;
  stopStream: () => void;
}

/**
 * Which run is this page showing?
 *
 * One server can serve many runs, so the run name travels in the URL rather than
 * in server state. That keeps two browser tabs on two different runs from
 * fighting over a shared "active run", and makes a run's URL shareable.
 * Absent, the server falls back to whichever run it was started with.
 */
export function currentRun(): string | null {
  try {
    return new URLSearchParams(window.location.search).get("run");
  } catch {
    return null;
  }
}

/**
 * Should the page start driving on its own?
 *
 * The studio hands off with `?run=X&play=1`, so opening a finished job lands
 * you in the moving map rather than on a static overview with a play button to
 * find. `play=live` replays it over SSE instead, findings arriving as alerts.
 *
 * Autoplay is a URL parameter and not the default because the plain viewer URL
 * is also how you go back to study one finding, and that must stay still.
 */
export type Autoplay = "off" | "scrub" | "live";

export function autoplayMode(): Autoplay {
  try {
    const v = (new URLSearchParams(window.location.search).get("play") || "").toLowerCase();
    if (v === "live") return "live";
    if (v === "1" || v === "true" || v === "yes") return "scrub";
    return "off";
  } catch {
    return "off";
  }
}

/** Append ?run= to an API path when the page is scoped to a named run. */
export function apiUrl(path: string): string {
  const run = currentRun();
  if (!run) return path;
  return path + (path.includes("?") ? "&" : "?") + "run=" + encodeURIComponent(run);
}

async function getJSON<T>(url: string, fallback: T): Promise<T> {
  try {
    const r = await fetch(url);
    if (!r.ok) return fallback;
    return (await r.json()) as T;
  } catch {
    return fallback;
  }
}

let eventSource: EventSource | null = null;

export const useStore = create<State>((set, get) => ({
  loading: true,
  error: null,

  manifest: null,
  findings: [],
  segments: [],
  frames: [],
  twin: { buildings: [], roads: [] },
  coverage: [],
  classMap: {},

  selected: null,
  hovered: null,
  classFilter: new Set<string>(),
  basemap: null,
  layers: {
    satellite: true,
    cloud: true,
    buildings: true,
    heatmap: true,
    markers: true,
    route: true,
  },
  layout: "overlay",

  playT: 0,
  playing: false,
  follow: true,
  streamed: new Set<string>(),
  streaming: false,
  alerts: [],

  async loadRun() {
    set({ loading: true, error: null });

    const manifest = await getJSON<Manifest | null>(apiUrl("/api/manifest"), null);
    if (!manifest || (manifest as unknown as { error?: string }).error) {
      set({
        loading: false,
        error:
          "No run found. Generate one first:\n\n" +
          "  uv run python scripts/make_sample.py\n" +
          "  uv run pos run --video samples/road/road.mp4 \\\n" +
          "      --gpx samples/road/track.gpx \\\n" +
          "      --truth samples/road/truth.json --out run\n" +
          "  uv run pos serve --run run",
      });
      return;
    }

    const [findings, segments, frames, twin, coverage, basemap] = await Promise.all([
      getJSON<Finding[]>(apiUrl("/api/findings"), []),
      getJSON<Segment[]>(apiUrl("/api/segments"), []),
      getJSON<Frame[]>(apiUrl("/api/frames"), []),
      getJSON<Twin>(apiUrl("/api/twin"), { buildings: [], roads: [] }),
      getJSON<CoverageRow[]>(apiUrl("/api/coverage"), []),
      // Optional: the endpoint answers {available:false} rather than 404ing, so
      // a run with no imagery is a normal state, not an error.
      getJSON<Basemap>(apiUrl("/api/basemap"), { available: false }),
    ]);

    const classMap: Record<string, ClassSpec> = {};
    for (const c of manifest.classes ?? []) classMap[c.key] = c;

    const auto = autoplayMode();

    set({
      loading: false,
      manifest,
      findings,
      segments,
      frames,
      twin,
      coverage,
      basemap,
      classMap,
      // Rewind before autoplaying: the default resting state is the END of the
      // run, with every finding revealed, which is right for studying a result
      // and wrong for driving it.
      playT: auto === "off" ? manifest.duration_sec || 0 : 0,
      playing: auto === "scrub",
      follow: auto === "off" ? get().follow : true,
    });

    // Started after the state above lands, because startStream() resets the
    // reveal set and opens the SSE connection -- it has to act on a store that
    // already knows the run.
    if (auto === "live") get().startStream(1);
  },

  select: (f) => set({ selected: f }),
  hover: (id) => set({ hovered: id }),

  toggleClass: (key) =>
    set((s) => {
      const next = new Set(s.classFilter);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return { classFilter: next };
    }),

  clearFilter: () => set({ classFilter: new Set<string>() }),

  toggleLayer: (key) =>
    set((s) => ({ layers: { ...s.layers, [key]: !s.layers[key] } })),

  setLayout: (l) => set({ layout: l }),

  setPlayT: (t) => set({ playT: t }),
  setPlaying: (p) => set({ playing: p }),
  toggleFollow: () => set((s) => ({ follow: !s.follow })),

  startStream(speed = 1) {
    // 1x by default so the video panel stays in step. The video plays at its
    // natural rate, so replaying findings any faster would leave the footage
    // lagging behind the markers -- which reads as a bug, not a feature.
    get().stopStream();
    // Reveal nothing, then let the stream fill the map in. This is the
    // "driving now" mode -- the same endpoint replaying a saved run.
    set({ streamed: new Set<string>(), streaming: true, alerts: [], playT: 0 });

    const es = new EventSource(apiUrl(`/stream?speed=${speed}`));
    eventSource = es;

    es.onmessage = (ev) => {
      let msg: StreamEvent;
      try {
        msg = JSON.parse(ev.data) as StreamEvent;
      } catch {
        return;
      }

      if (msg.type === "done") {
        set({ streaming: false });
        es.close();
        eventSource = null;
        return;
      }

      const f = msg.finding;
      if (!f) return;

      set((s) => {
        const streamed = new Set(s.streamed);
        streamed.add(f.finding_id);
        const alerts = f.alert
          ? [
              { id: f.finding_id, label: f.label, severity: f.severity, t: msg.t_sec },
              ...s.alerts,
            ].slice(0, 6)
          : s.alerts;
        return { streamed, alerts, playT: msg.t_sec };
      });
    };

    es.onerror = () => {
      set({ streaming: false });
      es.close();
      eventSource = null;
    };
  },

  stopStream() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    set({ streaming: false });
  },
}));

/**
 * Findings currently visible, honouring the class filter and playback head.
 *
 * This MUST be a hook with its own useMemo rather than a plain zustand
 * selector. A selector that builds a new array runs afoul of zustand's
 * reference equality check -- every render returns a fresh array, which looks
 * like a state change, which triggers another render. That is an infinite loop
 * (React error #185). Selecting the individual fields is safe because each one
 * is a stable reference between updates.
 */
export function useVisibleFindings(): Finding[] {
  const findings = useStore((s) => s.findings);
  const classFilter = useStore((s) => s.classFilter);
  const streaming = useStore((s) => s.streaming);
  const streamed = useStore((s) => s.streamed);
  const playT = useStore((s) => s.playT);

  return useMemo(
    () =>
      findings.filter((f) => {
        if (classFilter.size > 0 && !classFilter.has(f.cls)) return false;
        if (streaming) return streamed.has(f.finding_id);
        return f.t_sec <= playT + 1e-6;
      }),
    [findings, classFilter, streaming, streamed, playT]
  );
}

/**
 * Colour and label lookups take the classMap directly rather than the whole
 * State, so callers can subscribe to just `s.classMap` instead of the entire
 * store. Subscribing to everything re-renders the 3D scene on every timeline
 * tick, which is a real frame-rate cost during playback.
 */
export type ClassMap = Record<string, ClassSpec>;

export function colorOf(classMap: ClassMap, cls: string): string {
  return classMap[cls]?.color ?? "#f59e0b";
}

export function labelOf(classMap: ClassMap, cls: string): string {
  return classMap[cls]?.label ?? cls;
}

/** Inferred-from-absence classes get their own rendering: no box, a "missing" tag. */
export function isAbsence(classMap: ClassMap, cls: string): boolean {
  return classMap[cls]?.absence === true;
}

export function isAsset(classMap: ClassMap, cls: string): boolean {
  // weight 0 means we record it as an asset, not penalise it as a defect.
  return (classMap[cls]?.weight ?? 1) === 0;
}
