import { useEffect, useRef } from "react";
import { apiUrl, colorOf, isAbsence, useStore, useVisibleFindings } from "./store";
import type { LayerKey } from "./store";

/**
 * Left panel (score + legend + layers), bottom timeline, and the alert stack.
 *
 * The legend doubles as the class filter. The taxonomy arrives from the domain
 * YAML via /api/manifest, so switching domain changes this UI with no code
 * change here.
 */

function gradeColor(index: number): string {
  if (index >= 90) return "#22c55e";
  if (index >= 75) return "#84cc16";
  if (index >= 60) return "#eab308";
  if (index >= 45) return "#f59e0b";
  if (index >= 30) return "#ea580c";
  return "#dc2626";
}

// --------------------------------------------------------------------------

export function ScorePanel() {
  const manifest = useStore((s) => s.manifest);
  const classMap = useStore((s) => s.classMap);
  const classFilter = useStore((s) => s.classFilter);
  const toggleClass = useStore((s) => s.toggleClass);
  const clearFilter = useStore((s) => s.clearFilter);
  const layers = useStore((s) => s.layers);
  const toggleLayer = useStore((s) => s.toggleLayer);
  const basemap = useStore((s) => s.basemap);
  const findings = useStore((s) => s.findings);
  const coverage = useStore((s) => s.coverage);
  const select = useStore((s) => s.select);

  if (!manifest) return null;
  const summary = manifest.summary;

  // Only show classes that actually occurred, most frequent first.
  const present = Object.entries(summary?.counts ?? {}).sort((a, b) => b[1] - a[1]);

  const layerList: [LayerKey, string][] = [
    ["satellite", "Satellite imagery"],
    ["heatmap", "Quality heatmap"],
    ["markers", "Findings"],
    ["buildings", "OSM buildings"],
    ["route", "Route line"],
    ["cloud", "Point cloud"],
  ];

  return (
    <aside className="panel panel-left">
      <div className="brand">
        <div className="brand-dot" />
        <div>
          <div className="brand-name">PhysicalOS</div>
          <div className="muted small">the physical world tells the truth</div>
        </div>
      </div>

      <div className="domain">{manifest.domain_label}</div>

      {summary && (
        <div className="score">
          <div
            className="score-ring"
            style={{ borderColor: gradeColor(summary.quality_index) }}
          >
            <div className="score-num">{summary.quality_index.toFixed(0)}</div>
            <div
              className="score-grade"
              style={{ color: gradeColor(summary.quality_index) }}
            >
              {summary.grade}
            </div>
          </div>
          <div className="score-meta">
            <div className="score-label">{manifest.index_name}</div>
            <div className="muted small">
              {summary.total_findings} findings over {summary.route_length_m.toFixed(0)} m
            </div>
            <div className="muted small">
              {manifest.n_frames} keyframes · {manifest.n_detections} detections
            </div>
          </div>
        </div>
      )}

      <div className="section-head">
        <span>Findings by type</span>
        {classFilter.size > 0 && (
          <button className="link" onClick={clearFilter}>
            clear
          </button>
        )}
      </div>

      <ul className="legend">
        {present.map(([cls, n]) => {
          const spec = classMap[cls];
          const active = classFilter.size === 0 || classFilter.has(cls);
          return (
            <li key={cls}>
              <button
                className={active ? "legend-row" : "legend-row dim"}
                onClick={() => toggleClass(cls)}
                title="Filter to this type"
              >
                <span className="swatch" style={{ background: colorOf(classMap, cls) }} />
                <span className="legend-label">{spec?.label ?? cls}</span>
                {isAbsence(classMap, cls) && (
                  <span className="missing-tag" title="Inferred from missing coverage">
                    MISSING
                  </span>
                )}
                {spec?.alert && (
                  <span className="bang" title="Alert class">
                    !
                  </span>
                )}
                <span className="legend-n">{n}</span>
              </button>
            </li>
          );
        })}
        {present.length === 0 && (
          <li className="muted small">No findings in this run.</li>
        )}
      </ul>

      {coverage.length > 0 && (
        <>
          <div className="section-head">
            <span>Asset coverage</span>
          </div>
          <ul className="coverage">
            {coverage.map((c) => {
              const bad = c.found === 0 || c.gaps > 0;
              const gapFinding = findings.find((f) => f.cls === c.absence_key);
              return (
                <li key={c.asset}>
                  <button
                    className={bad ? "cov-row bad" : "cov-row"}
                    onClick={() => gapFinding && select(gapFinding)}
                    title={
                      bad
                        ? "Click to see the stretch where it is missing"
                        : "Adequately covered"
                    }
                  >
                    <span className="cov-icon">{bad ? "✕" : "✓"}</span>
                    <span className="legend-label">{c.asset_label}</span>
                    <span className="cov-val">
                      {c.found === 0
                        ? "none found"
                        : `${c.found} · ${c.per_km}/km`}
                    </span>
                  </button>
                  {bad && (
                    <div className="cov-note">
                      {c.absence_label} — no detection over {c.route_m} m
                      (threshold {c.min_gap_m} m)
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </>
      )}

      <div className="section-head">
        <span>Layers</span>
      </div>
      <ul className="layers">
        {layerList.map(([key, label]) => {
          const unavailable =
            (key === "cloud" && !manifest.has_pointcloud) ||
            (key === "satellite" && !basemap?.available);
          const hint =
            key === "satellite" && !basemap?.available
              ? " — run pos basemap"
              : " — none in run";
          return (
            <li key={key}>
              <label className={unavailable ? "dim" : ""}>
                <input
                  type="checkbox"
                  checked={layers[key] && !unavailable}
                  disabled={unavailable}
                  onChange={() => toggleLayer(key)}
                />
                {label}
                {unavailable && <span className="muted small">{hint}</span>}
              </label>
            </li>
          );
        })}
      </ul>

      {/* Imagery attribution is a licence condition, not decoration. It comes
          from basemap.json rather than being hardcoded, because it differs per
          provider. */}
      {basemap?.available && layers.satellite && (
        <div className="attrib muted small">
          {basemap.attribution}
          {basemap.kind === "street" && " · street map, not satellite"}
        </div>
      )}

      <div className="section-head">
        <span>Export</span>
      </div>
      <ul className="exports">
        <li>
          <a href={apiUrl("/api/kml")} download>
            ⬇ Google Earth (.kmz)
          </a>
          <div className="muted small">
            Findings, evidence photos and the drive on a time slider. Opens in
            Google Earth Pro, whose Movie Maker records it to MP4.
          </div>
        </li>
        <li>
          {/* Two links on purpose: the plain one opens in the browser's PDF
              viewer, which is what you want for a quick look, and ?download=1
              flips Content-Disposition to attachment for filing or emailing. */}
          <a href={apiUrl("/api/report.pdf")} target="_blank" rel="noreferrer">
            ▤ Open PDF report
          </a>{" "}
          <a href={apiUrl("/api/report.pdf?download=1")} download>
            ⬇ save
          </a>
          <div className="muted small">
            Cover with the score and asset coverage, segments worst-first, then a
            page per finding with its photo and reasoning. Built on first request,
            so the first open takes a few seconds.
          </div>
        </li>
      </ul>

      <div className="section-head">
        <span>Worst findings</span>
      </div>
      <ul className="worst">
        {[...findings]
          .sort((a, b) => b.severity - a.severity || b.confidence - a.confidence)
          .slice(0, 5)
          .map((f) => (
            <li key={f.finding_id}>
              <button className="worst-row" onClick={() => select(f)}>
                <span className="swatch" style={{ background: colorOf(classMap, f.cls) }} />
                <span className="legend-label">{f.label}</span>
                <span className="sev">{f.severity}</span>
              </button>
            </li>
          ))}
      </ul>

      <div className="backend muted small">
        perception: <strong>{manifest.backend}</strong>
        {manifest.backend === "mock" && " (offline fixtures)"}
      </div>
    </aside>
  );
}

// --------------------------------------------------------------------------

export function Timeline() {
  const manifest = useStore((s) => s.manifest);
  const playT = useStore((s) => s.playT);
  const setPlayT = useStore((s) => s.setPlayT);
  const playing = useStore((s) => s.playing);
  const setPlaying = useStore((s) => s.setPlaying);
  const streaming = useStore((s) => s.streaming);
  const startStream = useStore((s) => s.startStream);
  const stopStream = useStore((s) => s.stopStream);
  const follow = useStore((s) => s.follow);
  const toggleFollow = useStore((s) => s.toggleFollow);
  const visible = useVisibleFindings();
  const layout = useStore((s) => s.layout);
  const setLayout = useStore((s) => s.setLayout);
  const hasVideo = useStore((s) => s.manifest?.has_video !== false);

  const raf = useRef<number | null>(null);
  const last = useRef<number>(0);
  const duration = manifest?.duration_sec ?? 0;

  // Local scrub playback, separate from the SSE stream: this only moves the
  // reveal head over already-loaded findings, so it can also run backwards.
  useEffect(() => {
    if (!playing || streaming) {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
      raf.current = null;
      return;
    }
    last.current = performance.now();

    const tick = (now: number) => {
      const dt = (now - last.current) / 1000;
      last.current = now;
      const next = useStore.getState().playT + dt; // 1x, matching the video
      if (next >= duration) {
        setPlayT(duration);
        setPlaying(false);
        return;
      }
      setPlayT(next);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);

    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
    };
  }, [playing, streaming, duration, setPlayT, setPlaying]);

  if (!manifest) return null;

  return (
    <div className="timeline">
      <button
        className="btn"
        onClick={() => {
          if (streaming) return;
          if (playT >= duration) setPlayT(0);
          setPlaying(!playing);
        }}
        disabled={streaming}
      >
        {playing ? "❚❚" : "▶"}
      </button>

      <input
        className="scrub"
        type="range"
        min={0}
        max={Math.max(duration, 0.1)}
        step={0.1}
        value={Math.min(playT, duration)}
        disabled={streaming}
        onChange={(e) => {
          setPlaying(false);
          setPlayT(parseFloat(e.target.value));
        }}
      />

      <div className="tcount mono">
        {Math.min(playT, duration).toFixed(1)}s / {duration.toFixed(0)}s
        <span className="muted"> · {visible.length} shown</span>
      </div>

      <button
        className={follow ? "btn btn-follow on" : "btn btn-follow"}
        onClick={toggleFollow}
        title={
          follow
            ? "Camera is following the drive — click to hold this view"
            : "Follow the drive along the route as it plays"
        }
      >
        {follow ? "⦿ following" : "⦾ follow"}
      </button>

      {hasVideo && (
        <button
          className="btn btn-layout"
          onClick={() => setLayout(layout === "split" ? "overlay" : "split")}
          title={
            layout === "split"
              ? "Overlay the video on the map"
              : "Show video and 3D map side by side"
          }
        >
          {layout === "split" ? "▣ overlay" : "◫ side by side"}
        </button>
      )}

      <button
        className={streaming ? "btn btn-live on" : "btn btn-live"}
        onClick={() => (streaming ? stopStream() : startStream(1))}
        title="Replay the drive as a live feed over SSE, in real time"
      >
        {streaming ? "◼ stop live" : "◉ live drive"}
      </button>
    </div>
  );
}

// --------------------------------------------------------------------------

export function AlertStack() {
  const alerts = useStore((s) => s.alerts);
  const findings = useStore((s) => s.findings);
  const select = useStore((s) => s.select);
  const streaming = useStore((s) => s.streaming);

  if (!alerts.length) return null;

  return (
    <div className="alerts">
      {streaming && <div className="alerts-head">● LIVE</div>}
      {alerts.map((a) => (
        <button
          key={a.id}
          className="alert"
          onClick={() => {
            const f = findings.find((x) => x.finding_id === a.id);
            if (f) select(f);
          }}
        >
          <span className="alert-sev">SEV {a.severity}</span>
          <span className="alert-label">{a.label}</span>
          <span className="muted small mono">{a.t.toFixed(1)}s</span>
        </button>
      ))}
    </div>
  );
}
