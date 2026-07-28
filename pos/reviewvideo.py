"""Render the drive back as an annotated video, so a human can watch the model work.

WHY THIS EXISTS
Everything else this pipeline produces is either abstract or static: markers on a
3D map, a PDF, single evidence crops. None of them answer the question a reviewer
actually asks -- "was that really a pothole, and did you see it where you say you
did?" -- because none of them show the defect arriving, being tracked, and passing
under the camera. This renders exactly that.

WHY KEYFRAMES AND NOT THE SOURCE VIDEO
The run's detections are keyed to keyframes by `frame_id`, and keyframes are what
the detector was actually shown. Re-decoding the source clip would give smoother
motion and a lie: most of those frames were never inspected, so a defect would
appear to be missed for a second at a time. Playback speed stays honest instead --
the output fps comes from the median gap between keyframes, so one second of video
is one second of driving.

TRACKING IS WHAT MAKES IT READABLE
Per-frame boxes flicker: a pothole seen five times is five unrelated rectangles,
which reads as instability rather than as one object being approached. ByteTrack
gives it one identity and TraceAnnotator draws where it has been, so the eye
follows a thing rather than a blink. Both the tracker and the smoother are
STATEFUL and are built once per render, never shared between runs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv

from .config import DomainConfig
from .schema import Detection
from .svbridge import class_palette, to_sv

# The road overlay is context, not a finding. At full strength it competes with
# the defect boxes drawn on top of it, which are the point of the frame.
ROAD_OPACITY = 0.28
ROAD_COLOR = sv.Color(56, 189, 248)  # the viewer's ego/trail cyan, so the two agree

# Metre-spaced sampling on a slow vehicle can leave 4 s between keyframes, and a
# 0.25 fps video is unwatchable. The honest rate is clamped, not trusted.
FPS_MIN = 2.0
FPS_MAX = 30.0
FPS_FALLBACK = 6.0

HUD_BG = (16, 20, 26)
HUD_FG = (235, 240, 246)

# Keyframes are segmented one at a time and independently, so the mask boundary
# jitters by a few pixels every frame and the overlay boils. Averaging the last
# N masks and re-thresholding costs nothing and removes it. 5 at ~3 fps is about
# 1.5 s of memory: long enough to kill the flicker, short enough that the mask
# still keeps up when the road genuinely changes shape at a junction.
SMOOTH_FRAMES = 5
# A pixel has to be road in more than half the remembered frames to be drawn.
# Lower and the mask smears forward over whatever the road just left.
SMOOTH_MAJORITY = 0.5


@dataclass(frozen=True)
class RenderResult:
    path: Path
    frames: int
    fps: float
    resolution: tuple[int, int]
    detections_drawn: int
    frames_with_road: int
    duration_s: float
    codec: str = "mp4v"


def _transcode_h264(src: Path, dst: Path, fps: float, crf: int) -> bool:
    """Re-encode to H.264, because a browser will not play what OpenCV writes.

    sv.VideoSink goes through cv2.VideoWriter, whose portable codec is MPEG-4
    Part 2 (`mp4v`). Chrome, Safari and the studio's own <video> tag all refuse
    it, and it is roughly 300 KB/frame at 1080p, so a 570-frame run lands near
    190 MB. ffmpeg is already a hard dependency of ingest, so the fix is a
    transcode rather than a new library. Returns False if ffmpeg is missing, in
    which case the caller keeps the playable-in-VLC original rather than nothing.
    """
    if shutil.which("ffmpeg") is None:
        return False
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            # yuv420p is the pixel format every browser decoder agrees on;
            # without it ffmpeg may pick yuv444p and Safari shows a black frame.
            "-pix_fmt", "yuv420p",
            # Puts the index at the front so the video starts playing before it
            # has fully downloaded -- the difference between a link that works
            # and one that hangs on a 190 MB file.
            "-movflags", "+faststart",
            # No -r here. Forcing an output rate makes ffmpeg re-time the stream
            # against the source's integer fps and drop a frame to fit (measured:
            # 30 in, 29 out). The input already carries the rate the sink wrote.
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and dst.exists()


def _load_frames(run: Path) -> list[dict]:
    frames = json.loads((run / "frames.json").read_text())
    return sorted(frames, key=lambda f: f["t_sec"])


def _load_detections(run: Path) -> dict[str, list[Detection]]:
    """Group the run's detections by frame, so each keyframe draws its own.

    Detections, not findings, and deliberately: findings are deduplicated across
    frames and carry one position for the whole cluster, so drawing them per
    frame would paint the same box onto frames where nothing was seen. This file
    is what the detector actually emitted, frame by frame.
    """
    path = run / "detections.ndjson"
    out: dict[str, list[Detection]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        det = Detection(**json.loads(line))
        out.setdefault(det.frame_id, []).append(det)
    return out


def _fps_from_keyframes(frames: list[dict], stride: int) -> float:
    """Playback rate that makes one video second equal one driving second."""
    times = [f["t_sec"] for f in frames][::stride]
    if len(times) < 2:
        return FPS_FALLBACK
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not gaps:
        return FPS_FALLBACK
    median_gap = sorted(gaps)[len(gaps) // 2]
    if median_gap <= 0:
        return FPS_FALLBACK
    return max(FPS_MIN, min(FPS_MAX, 1.0 / median_gap))


def _road_detections(mask: np.ndarray, w: int, h: int) -> sv.Detections:
    """Wrap a boolean road mask as a one-row sv.Detections for MaskAnnotator.

    sv annotators consume Detections and nothing else, so the mask travels as a
    single full-frame box carrying the mask array. It gets its own single-colour
    annotator rather than the class palette -- otherwise the road would be tinted
    with whatever defect class happens to sit at index 0.
    """
    return sv.Detections(
        xyxy=np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32),
        mask=mask[None, ...].astype(bool),
        class_id=np.array([0], dtype=int),
    )


def _draw_hud(scene: np.ndarray, frame: dict, counts: dict[str, int], road: bool) -> np.ndarray:
    import cv2

    h, w = scene.shape[:2]
    pad = 10
    lines = [
        f"t={frame['t_sec']:7.2f}s   frame {frame['frame_id']}",
        f"{frame['lat']:.6f}, {frame['lon']:.6f}   hdg {frame.get('heading_deg', 0):.0f}",
        "road: segmented" if road else "road: none found in this frame",
    ]
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        lines.append("sightings: " + "  ".join(f"{k} {v}" for k, v in top))

    box_h = pad * 2 + 22 * len(lines)
    box_w = min(w - 2 * pad, 760)
    overlay = scene.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.62, scene, 0.38, 0, scene)

    y = pad + 24
    for text in lines:
        cv2.putText(scene, text, (pad + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_FG, 1, cv2.LINE_AA)
        y += 22
    return scene


def render_review(
    run: Path,
    out: Path,
    domain: DomainConfig | None = None,
    stride: int = 1,
    limit: int = 0,
    segmenter=None,
    smooth_masks: int = SMOOTH_FRAMES,
    h264: bool = True,
    crf: int = 23,
) -> RenderResult:
    """Render `run` to an annotated MP4 at `out`.

    `segmenter` is injected rather than built here, so the caller decides whether
    to pay for road segmentation at all and a caller that has already segmented
    the run can reuse its loaded session. None means "no road overlay".
    """
    import cv2

    run = Path(run)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if domain is None:
        manifest = json.loads((run / "manifest.json").read_text())
        domain = DomainConfig.load(manifest.get("domain", "road"))

    all_frames = _load_frames(run)
    frames = all_frames[:: max(stride, 1)]
    if limit > 0:
        frames = frames[:limit]
    if not frames:
        raise ValueError(f"{run}/frames.json yielded no keyframes to render")

    by_frame = _load_detections(run)
    # Rate from the STRIDED sequence, so --stride 5 plays at 5x wall-clock rather
    # than silently pretending the drive took a fifth of the time.
    fps = _fps_from_keyframes(all_frames, max(stride, 1))
    w, h = int(frames[0]["width"]), int(frames[0]["height"])

    palette = class_palette(domain)
    box_annotator = sv.BoxAnnotator(color=palette, color_lookup=sv.ColorLookup.CLASS, thickness=3)
    label_annotator = sv.LabelAnnotator(
        color=palette, color_lookup=sv.ColorLookup.CLASS, text_scale=0.5, text_padding=4
    )
    # TRACK lookup, not CLASS: a trace is one object's path, so its colour must
    # follow the identity even where two defects of the same class overlap.
    trace_annotator = sv.TraceAnnotator(
        color_lookup=sv.ColorLookup.TRACK, thickness=2, trace_length=24
    )
    mask_annotator = sv.MaskAnnotator(color=ROAD_COLOR, opacity=ROAD_OPACITY)

    # Stateful, one set per render. Sharing either across runs leaks tracker ids
    # and smoothing history from the previous drive into this one.
    tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps))))
    smoother = sv.DetectionsSmoother()

    counts: dict[str, int] = {}
    seen_tracks: set[int] = set()
    drawn = 0
    frames_with_road = 0
    written = 0

    # Ring buffer of recent masks for temporal smoothing. Bool arrays, summed on
    # demand: five 1080p masks is ~10 MB, where keeping them as float32 for a
    # running mean would be four times that for the same answer.
    recent: deque[np.ndarray] = deque(maxlen=max(1, smooth_masks))

    # Render to a temp file when transcoding, so a half-written H.264 file can
    # never masquerade as a finished render at the caller's path.
    raw_path = out.with_suffix(".raw.mp4") if h264 else out

    info = sv.VideoInfo(width=w, height=h, fps=max(1, int(round(fps))), total_frames=len(frames))
    with sv.VideoSink(target_path=str(raw_path), video_info=info) as sink:
        for frame in frames:
            img_path = run / frame["path"]
            scene = cv2.imread(str(img_path))
            if scene is None:
                continue
            fh, fw = scene.shape[:2]

            mask = None
            if segmenter is not None:
                try:
                    mask = segmenter.mask(img_path, (fw, fh))
                except Exception:  # noqa: BLE001 - one bad frame must not kill the render
                    mask = None
            has_road = mask is not None and bool(mask.any())
            if has_road:
                frames_with_road += 1

            # Smooth over the recent past, not just this frame. A frame where the
            # segmenter found nothing (measured: it does that on crowded market
            # scenes) contributes an empty mask rather than being skipped --
            # skipping would let the previous mask persist over a stretch with no
            # road, which is a worse lie than a momentary gap.
            if mask is not None:
                recent.append(mask)
            elif recent:
                recent.append(np.zeros_like(recent[-1]))

            draw_mask = None
            if recent:
                stacked = np.sum(recent, axis=0)
                draw_mask = stacked > (len(recent) * SMOOTH_MAJORITY)
                if not draw_mask.any():
                    draw_mask = None

            if draw_mask is not None:
                scene = mask_annotator.annotate(
                    scene=scene, detections=_road_detections(draw_mask, fw, fh)
                )

            # Per-frame dimensions, never frames[0]: the bridge cannot tell a
            # wrong frame size from a right one, it just misplaces every box.
            sv_dets = to_sv(by_frame.get(frame["frame_id"], []), fw, fh, domain)

            # THE BOXES ARE DRAWN FROM THE RAW DETECTIONS, NOT THE TRACKER, and
            # that is not an oversight. ByteTrack only emits a track once it has
            # been confirmed over consecutive frames; these keyframes are spaced
            # by METRES of travel, so the same defect seldom lands in two
            # consecutive ones with enough overlap to confirm. Measured on
            # runs/POC-1: 31 detections in, 0 out, including 0.90-confidence
            # rows. Feeding the annotators tracker output would silently show
            # fewer defects than the pipeline found, which is the one thing a
            # review video must never do.
            tracked = tracker.update_with_detections(sv_dets)
            if len(tracked):
                # Smoothing keys on tracker_id, so it only has meaning here.
                tracked = smoother.update_with_detections(tracked)
                scene = trace_annotator.annotate(scene=scene, detections=tracked)
                if tracked.tracker_id is not None:
                    for tid in tracked.tracker_id:
                        seen_tracks.add(int(tid))

            if len(sv_dets):
                names = list(sv_dets.data.get("class_name", []))
                confs = (
                    sv_dets.confidence
                    if sv_dets.confidence is not None
                    else np.ones(len(sv_dets), dtype=float)
                )
                labels = [f"{n} {c:.2f}" for n, c in zip(names, confs)]
                scene = box_annotator.annotate(scene=scene, detections=sv_dets)
                scene = label_annotator.annotate(scene=scene, detections=sv_dets, labels=labels)
                drawn += len(sv_dets)

                # Sightings, not objects. Deduplicating to objects needs stable
                # identities, and the tracker cannot supply them at this frame
                # spacing, so the HUD says "sightings" rather than implying a
                # count of distinct defects it has not earned.
                for cls in sv_dets.data.get("cls", []):
                    counts[cls] = counts.get(cls, 0) + 1

            scene = _draw_hud(scene, frame, counts, has_road)
            sink.write_frame(scene)
            written += 1

    codec = "mp4v"
    if h264:
        if _transcode_h264(raw_path, out, fps, crf):
            codec = "h264"
            raw_path.unlink(missing_ok=True)
        else:
            # No ffmpeg. Keep what was rendered rather than losing the run, but
            # do not claim a codec the file does not have -- the caller prints
            # this, and a browser is about to refuse the file.
            raw_path.replace(out)

    return RenderResult(
        path=out,
        frames=written,
        fps=fps,
        resolution=(w, h),
        detections_drawn=drawn,
        frames_with_road=frames_with_road,
        duration_s=written / fps if fps else 0.0,
        codec=codec,
    )
