import { useEffect } from "react";
import { Scene } from "./Scene";
import { EvidencePanel } from "./EvidencePanel";
import { AlertStack, ScorePanel, Timeline } from "./panels";
import { VideoPanel } from "./VideoPanel";
import { useStore } from "./store";

/**
 * Layout: the 3D twin fills the window, panels float over it.
 *
 * The 3D view is the subject, not a widget in a dashboard -- so panels are
 * overlaid rather than laid out beside it, and the canvas never resizes when a
 * panel opens or closes.
 */
export function App() {
  const loading = useStore((s) => s.loading);
  const error = useStore((s) => s.error);
  const manifest = useStore((s) => s.manifest);
  const layout = useStore((s) => s.layout);
  const loadRun = useStore((s) => s.loadRun);

  useEffect(() => {
    void loadRun();
  }, [loadRun]);

  if (loading) {
    return (
      <div className="boot">
        <div className="boot-spinner" />
        <div className="boot-text">
          <div className="brand-name" style={{ fontSize: '18px', marginBottom: '4px' }}>PhysicalOS</div>
          <div className="muted small">Loading run…</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="boot boot-error">
        <h1>PhysicalOS</h1>
        <p className="muted">No run is being served on this port.</p>
        <pre>{error}</pre>
      </div>
    );
  }

  return (
    // `layout` drives everything through CSS. In overlay mode .stage fills the
    // window and .rail's children keep their own absolute positions; in split
    // mode .stage shrinks and .rail becomes a real column beside it. Doing it in
    // CSS rather than by swapping components means the R3F canvas is never
    // unmounted -- it just resizes, so the camera and scene survive the toggle.
    <div className={`app layout-${layout}`}>
      <div className="stage">
        <Scene />
      </div>
      <ScorePanel />
      <div className="rail">
        <VideoPanel />
        <EvidencePanel />
      </div>
      <AlertStack />
      <Timeline />
      {manifest && (
        <div className="runtag muted small mono">
          {manifest.run_id} · {manifest.video}
        </div>
      )}
    </div>
  );
}
