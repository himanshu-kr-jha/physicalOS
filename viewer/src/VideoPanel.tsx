import { useEffect, useRef, useState } from "react";
import { apiUrl, useStore } from "./store";

/**
 * The source clip, played beside the 3D twin and kept in step with the timeline.
 *
 * This is what makes live mode legible: markers appearing on a map are abstract
 * until you can see the road they came from at the same moment. Both clocks are
 * the same -- playT and Frame.t_sec are both seconds of video time -- so syncing
 * is just keeping video.currentTime near playT.
 *
 * Two sync modes, because seeking and playing fight each other:
 *   - live/playing : let the element play, and correct only when drift exceeds
 *                    DRIFT_S. Assigning currentTime every tick restarts the
 *                    decoder and reads as a stutter.
 *   - scrubbing    : seek directly; there is no playback to preserve.
 */

const DRIFT_S = 0.45;

export function VideoPanel() {
  const manifest = useStore((s) => s.manifest);
  const playT = useStore((s) => s.playT);
  const playing = useStore((s) => s.playing);
  const streaming = useStore((s) => s.streaming);

  const ref = useRef<HTMLVideoElement | null>(null);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState(true);

  const active = playing || streaming;

  // Keep the element in step with the timeline.
  useEffect(() => {
    const v = ref.current;
    if (!v || failed) return;

    if (Math.abs(v.currentTime - playT) > DRIFT_S) {
      try {
        v.currentTime = playT;
      } catch {
        /* seeking before metadata loads throws; the next tick retries */
      }
    }

    if (active && v.paused) void v.play().catch(() => undefined);
    if (!active && !v.paused) v.pause();
  }, [playT, active, failed]);

  // Always 1x. Real-time playback is what makes the footage readable -- you can
  // actually see the defect the marker refers to. The SSE stream and the local
  // scrub both run at 1x too, so all three clocks agree.
  useEffect(() => {
    const v = ref.current;
    if (v) v.playbackRate = 1;
  }, [streaming]);

  if (!manifest || manifest.has_video === false || failed) return null;

  if (!open) {
    return (
      <button className="vid-reopen" onClick={() => setOpen(true)}>
        ▣ show video
      </button>
    );
  }

  return (
    <div className={active ? "vidpanel live" : "vidpanel"}>
      <div className="vid-head">
        <span>
          {streaming ? "● LIVE FEED" : "SOURCE VIDEO"}
          <span className="vid-rate"> 1×</span>
        </span>
        <button onClick={() => setOpen(false)} aria-label="Hide video">
          ×
        </button>
      </div>
      <video
        ref={ref}
        src={apiUrl("/api/video")}
        muted
        playsInline
        preload="metadata"
        onError={() => setFailed(true)}
      />
      <div className="vid-foot mono">
        {playT.toFixed(1)}s / {(manifest.duration_sec ?? 0).toFixed(0)}s
      </div>
    </div>
  );
}
