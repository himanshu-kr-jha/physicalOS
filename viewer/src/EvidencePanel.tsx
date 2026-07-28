import { useEffect, useState } from "react";
import { apiUrl, isAbsence, labelOf, useStore } from "./store";
import { BOX_SCALE } from "./types";
import type { Detection } from "./types";

/**
 * The point of the whole system: every claim on the map is one click from the
 * pixels that justify it.
 *
 * The box overlay is positioned in PERCENTAGES derived from BOX_SCALE, so it
 * scales with whatever size the image renders at and cannot drift out of sync
 * with the pipeline's convention (0..1000, origin top-left). This is the single
 * easiest thing in the project to get wrong, which is why the conversion lives
 * in exactly one place -- here.
 */

function pct(v: number): string {
  return `${(v / BOX_SCALE) * 100}%`;
}

function BoxOverlay({ det, color }: { det: Detection; color: string }) {
  const [x1, y1, x2, y2] = det.box;
  return (
    <div
      className="evbox"
      style={{
        left: pct(x1),
        top: pct(y1),
        width: pct(x2 - x1),
        height: pct(y2 - y1),
        borderColor: color,
        boxShadow: `0 0 0 1px rgba(0,0,0,.55), 0 0 18px ${color}55`,
      }}
    />
  );
}

/**
 * Index of the most convincing sighting: the one with the largest box.
 *
 * Evidence is stored in time order, so evidence[0] is the FIRST time the object
 * was seen -- which is the furthest away, and therefore the smallest and least
 * legible. A flat pothole 30 m down the road is a genuinely correct but useless
 * few-pixel sliver. Opening on the largest box shows the closest pass, which is
 * the frame a human inspector would actually want to look at.
 */
function bestSighting(evidence: Detection[]): number {
  let best = 0;
  let bestArea = -1;
  evidence.forEach((d, i) => {
    const [x1, y1, x2, y2] = d.box;
    const area = (x2 - x1) * (y2 - y1);
    if (area > bestArea) {
      bestArea = area;
      best = i;
    }
  });
  return best;
}

export function EvidencePanel() {
  const selected = useStore((s) => s.selected);
  const select = useStore((s) => s.select);
  const classMap = useStore((s) => s.classMap);
  const [idx, setIdx] = useState(0);

  // Jump to the clearest sighting whenever the finding changes.
  useEffect(() => {
    setIdx(selected ? bestSighting(selected.evidence ?? []) : 0);
  }, [selected]);

  if (!selected) {
    return (
      <aside className="panel panel-right panel-empty">
        <h2>Evidence</h2>
        <p className="muted">Click any marker, or a coloured road segment.</p>
        <p className="muted small">
          Every finding opens the actual frame it came from, with the model's
          bounding box and its stated reason.
        </p>
      </aside>
    );
  }

  const color = classMap[selected.cls]?.color ?? "#f59e0b";
  const absent = isAbsence(classMap, selected.cls);
  const evidence = selected.evidence ?? [];
  const det = evidence[Math.min(idx, Math.max(evidence.length - 1, 0))];

  return (
    <aside className="panel panel-right">
      <div className="panel-head">
        <div>
          <h2 style={{ color }}>{labelOf(classMap, selected.cls)}</h2>
          <div className="muted small">{selected.finding_id}</div>
        </div>
        <button className="x" onClick={() => select(null)} aria-label="Close">
          ×
        </button>
      </div>

      <div className="chips">
        <span className="chip" style={{ borderColor: color }}>
          severity {selected.severity}/5
        </span>
        <span className="chip">{(selected.confidence * 100).toFixed(0)}% confidence</span>
        <span className="chip">
          {evidence.length} sighting{evidence.length === 1 ? "" : "s"}
        </span>
      </div>

      {det ? (
        <>
          {absent && (
            <div className="absbanner">
              <strong>NOT DETECTED ANYWHERE ALONG THIS STRETCH</strong>
              <div>
                Inferred from coverage, not seen in a frame. The image below is
                simply the middle of the gap, so there is no box to draw.
              </div>
            </div>
          )}

          <div className="evwrap">
            <img
              src={apiUrl(`/frames/${det.frame_id}.jpg`)}
              alt={`Frame ${det.frame_id}`}
              draggable={false}
            />
            {/* An absence has no location within the frame, so drawing a box
                would assert something false. */}
            {!absent && <BoxOverlay det={det} color={color} />}
          </div>

          {evidence.length > 1 && (
            <div className="evnav">
              <button onClick={() => setIdx((i) => Math.max(0, i - 1))} disabled={idx === 0}>
                ‹
              </button>
              <span className="muted small">
                sighting {idx + 1} of {evidence.length} · frame {det.frame_id}
              </span>
              <button
                onClick={() => setIdx((i) => Math.min(evidence.length - 1, i + 1))}
                disabled={idx >= evidence.length - 1}
              >
                ›
              </button>
            </div>
          )}

          <blockquote className="reason">
            {det.evidence || "No reason given."}
          </blockquote>

          <dl className="facts">
            <dt>Position</dt>
            <dd className="mono">
              {selected.lat !== null && selected.lon !== null
                ? `${selected.lat.toFixed(6)}, ${selected.lon.toFixed(6)}`
                : "not localised"}
            </dd>

            <dt>Range at sighting</dt>
            <dd>
              {det.range_m !== null
                ? `${det.range_m.toFixed(1)} m ahead`
                : "above horizon — pinned to camera"}
            </dd>

            <dt>First seen</dt>
            <dd>{selected.t_sec.toFixed(1)} s into the drive</dd>

            {/* How well the position is actually known. A single ground-plane
                projection can only claim the blanket ±2–4 m; a triangulated fix
                has a real residual in metres, so say which this is rather than
                applying one disclaimer to everything. */}
            <dt>Position accuracy</dt>
            <dd>
              {selected.pos_method === "triangulated" ? (
                <>
                  <span style={{ color: "#7bd88f" }}>
                    ±{(selected.pos_residual_m ?? 0).toFixed(1)} m
                  </span>{" "}
                  triangulated from {selected.n_rays} rays
                  {selected.parallax_deg != null &&
                    ` (${selected.parallax_deg.toFixed(0)}° apart)`}
                </>
              ) : selected.pos_method === "camera" ? (
                "camera position — above horizon, never ranged"
              ) : (
                <>
                  ±2–4 m, single ground-plane projection
                  {(selected.n_rays ?? 0) > 1 &&
                    ` · ${selected.n_rays} rays too aligned to triangulate`}
                </>
              )}
            </dd>

            <dt>Box (0–1000, top-left)</dt>
            <dd className="mono">
              [{det.box.map((v) => Math.round(v)).join(", ")}]
              {det.anchor === "centre" && (
                <span className="muted"> · ranged from box centre</span>
              )}
            </dd>
          </dl>
        </>
      ) : (
        <p className="muted">This finding carries no frame evidence.</p>
      )}
    </aside>
  );
}
