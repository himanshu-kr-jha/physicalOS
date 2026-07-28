"""Turn a video + GPS track into georeferenced keyframes.

Every later stage depends on one thing being right here: each keyframe must
know where it was taken and which way the camera was pointing. Get the
heading wrong and findings land on the wrong side of the street.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from .geo import bearing_deg, haversine_m
from .schema import Frame


class IngestError(RuntimeError):
    pass


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise IngestError("ffmpeg not found on PATH. Install it: sudo apt install ffmpeg")


def _passthrough_args() -> list[str]:
    """Return the ffmpeg flag that disables frame-rate re-timing.

    ffmpeg >= 5.1 uses ``-fps_mode passthrough``; older builds (Ubuntu 22.04
    ships 4.4.x) only know ``-vsync 0``, which does the same thing.
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-fps_mode", "passthrough", "-f",
             "lavfi", "-i", "nullsrc=s=2x2:d=0", "-frames:v", "0",
             "-f", "null", "-"],
            capture_output=True, text=True,
        )
        if out.returncode == 0:
            return ["-fps_mode", "passthrough"]
    except Exception:  # noqa: BLE001
        pass
    return ["-vsync", "0"]


def probe_duration(video: Path) -> float:
    """Duration in seconds via ffprobe. 0.0 if it cannot be determined."""
    if shutil.which("ffprobe") is None:
        return 0.0
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def probe_fps(video: Path) -> float:
    """Source frame rate via ffprobe. 0.0 if it cannot be determined."""
    if shutil.which("ffprobe") is None:
        return 0.0
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    raw = out.stdout.strip()
    try:
        if "/" in raw:
            num, den = raw.split("/", 1)
            return float(num) / float(den) if float(den) else 0.0
        return float(raw)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_frame_times(video: Path) -> list[float]:
    """Presentation timestamp of every video frame, in seconds.

    This is the only trustworthy source of frame timing. `r_frame_rate` is a
    NOMINAL rate: on variable-frame-rate video -- which is what most phones
    record -- it reports the peak rate while the real frame spacing varies, so
    any time computed as index/rate drifts. Measured on a VFR test clip that
    drift reached 0.43 s, i.e. ~3.5 m of position error at 8 m/s.

    Returns [] if the timestamps cannot be read, so callers can fall back.
    """
    if shutil.which("ffprobe") is None:
        return []
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    times: list[float] = []
    for token in out.stdout.replace(",", " ").split():
        try:
            times.append(float(token))
        except ValueError:
            continue  # "N/A" on frames carrying no timestamp
    times.sort()
    return times


def extract_frames(
    video: Path, out_dir: Path, fps: float = 2.0
) -> tuple[list[Path], list[float]]:
    """Sample the video at ~`fps` and write JPEGs.

    Returns (paths, true_source_times_in_seconds).

    Two traps are avoided here. Both silently corrupt every finding's position
    rather than failing loudly, which is what makes them dangerous:

    1. We do NOT use ffmpeg's `fps` filter and then assume output frame n came
       from time n/fps. `fps` picks the source frame nearest the CENTRE of each
       output interval, so timestamps are off by up to half a sample period
       (measured: 0.20 s, ~1.6 m at 8 m/s). We select source frames by index.

    2. We do NOT compute those frames' times as index/r_frame_rate. On
       variable-frame-rate video that rate is nominal and real spacing varies
       (measured: 0.43 s, ~3.5 m at 8 m/s). We read each kept frame's actual
       presentation timestamp instead.
    """
    _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    src_fps = probe_fps(video)
    step = int(round(src_fps / fps)) if src_fps > 0 else 0

    if step >= 1:
        # Keep every `step`-th source frame. -fps_mode passthrough stops ffmpeg
        # re-timing the output and dropping or duplicating frames.
        args = ["-vf", f"select=not(mod(n\\,{step}))", *_passthrough_args()]
    else:
        # Unknown source rate: fall back to the fps filter.
        args = ["-vf", f"fps={fps}"]

    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video),
            *args,
            "-q:v", "3",
            str(out_dir / "%05d.jpg"),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise IngestError(f"ffmpeg failed:\n{proc.stderr}")

    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise IngestError(f"No frames extracted from {video}")

    times = _true_times(video, len(frames), step, src_fps, fps)
    return frames, times


def plan_spacing_indices(
    frame_times: list[float],
    track: list[TrackPoint],
    spacing_m: float,
    time_offset: float = 0.0,
) -> tuple[list[int], dict]:
    """Choose source frame indices ~`spacing_m` apart ALONG THE ROUTE.

    WHY DISTANCE AND NOT TIME
    `--fps` couples sampling to time, but everything downstream cares about
    distance. Measured on real footage: at 1 fps a 6.0 m/s vehicle leaves 6.4 m
    between keyframes and 153 of 158 objects are seen exactly ONCE. One look means
    no corroboration, nothing to track across frames, and no way to separate a
    real defect from a one-frame hallucination. The same 1 fps on a 2.0 m/s bike
    gives 2.6 m spacing and half the objects are seen twice or more.

    So the control that matters is metres, not seconds. Walk the GPS track and
    keep a frame each time the vehicle has moved `spacing_m` since the last one.
    Stationary stretches then contribute nothing, which is correct -- twenty
    frames of a stopped vehicle are twenty copies of one observation, and paying
    a VLM twenty times for it is waste.

    Returns (indices, stats).
    """
    if not frame_times or len(track) < 2:
        return list(range(len(frame_times))), {"reason": "no track"}

    keep: list[int] = []
    last: tuple[float, float] | None = None
    gaps: list[float] = []

    for i, t in enumerate(frame_times):
        lat, lon = interpolate(track, t + time_offset)
        if last is None:
            keep.append(i)
            last = (lat, lon)
            continue
        d = haversine_m(last[0], last[1], lat, lon)
        if d >= spacing_m:
            keep.append(i)
            gaps.append(d)
            last = (lat, lon)

    # Always keep the final frame: the tail of the route is data too.
    if keep and keep[-1] != len(frame_times) - 1:
        keep.append(len(frame_times) - 1)

    first = interpolate(track, frame_times[0] + time_offset)
    last_pt = interpolate(track, frame_times[-1] + time_offset)
    med = sorted(gaps)[len(gaps) // 2] if gaps else None

    return keep, {
        "source_frames": len(frame_times),
        "kept": len(keep),
        "straight_line_m": round(haversine_m(*first, *last_pt), 1),
        "requested_spacing_m": spacing_m,
        "achieved_median_m": round(med, 2) if med is not None else None,
        # If the source cannot supply the requested density, say so rather than
        # silently handing back coarser frames than were asked for.
        "source_limited": bool(med is not None and med > spacing_m * 1.5),
    }


def extract_frames_at(
    video: Path, out_dir: Path, indices: list[int], frame_times: list[float]
) -> tuple[list[Path], list[float]]:
    """Extract exactly the given source frame indices.

    Same discipline as extract_frames: select by INDEX, and take timestamps from
    the true presentation times rather than reconstructing them from an assumed
    rate. Both traps are documented on extract_frames, and both silently corrupt
    every finding's position instead of failing loudly.
    """
    _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    if not indices:
        raise IngestError("No frame indices to extract")

    # An explicit index set, not a modulo: the spacing is irregular by design,
    # because the vehicle's speed is.
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video),
            "-vf", f"select='{expr}'",
            *_passthrough_args(),
            "-q:v", "3",
            str(out_dir / "%05d.jpg"),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise IngestError(f"ffmpeg failed:\n{proc.stderr}")

    paths = sorted(out_dir.glob("*.jpg"))
    if not paths:
        raise IngestError(f"No frames extracted from {video}")

    times = [frame_times[i] for i in indices if i < len(frame_times)]
    # ffmpeg can emit one fewer than requested if an index lands past the end.
    if len(times) > len(paths):
        times = times[: len(paths)]
    return paths, times


def _true_times(
    video: Path, n_frames: int, step: int, src_fps: float, fps: float
) -> list[float]:
    """Timestamps of the kept frames, preferring measured PTS over nominal rate."""
    src_times = probe_frame_times(video) if step >= 1 else []

    if src_times:
        out: list[float] = []
        for i in range(n_frames):
            j = i * step
            if j < len(src_times):
                out.append(src_times[j])
            else:
                # Fewer timestamps than expected (truncated or odd container):
                # extrapolate at the nominal rate rather than dropping frames.
                out.append(j / (src_fps or fps))
        return out

    rate = src_fps if step >= 1 and src_fps > 0 else 0.0
    return [(i * step) / rate if rate > 0 else i / fps for i in range(n_frames)]


# --------------------------------------------------------------------------
# GPS track
# --------------------------------------------------------------------------


class TrackPoint:
    __slots__ = ("lat", "lon", "t")

    def __init__(self, lat: float, lon: float, t: float):
        self.lat, self.lon, self.t = lat, lon, t


def load_gpx(path: Path) -> tuple[list[TrackPoint], str | None]:
    """Parse a GPX file into points with seconds-since-track-start.

    Returns (points, iso_start_time_or_None).
    """
    import gpxpy

    with open(path) as fh:
        gpx = gpxpy.parse(fh)

    raw: list[tuple[float, float, object]] = [
        (p.latitude, p.longitude, p.time)
        for trk in gpx.tracks
        for seg in trk.segments
        for p in seg.points
    ]
    # Some devices only log waypoints, not tracks.
    if not raw:
        raw = [(w.latitude, w.longitude, w.time) for w in gpx.waypoints]
    if not raw:
        raise IngestError(f"No track points found in {path}")

    stamped = [r for r in raw if r[2] is not None]
    if stamped:
        t0 = stamped[0][2]
        pts = [
            TrackPoint(lat, lon, (t - t0).total_seconds())  # type: ignore[operator]
            for lat, lon, t in stamped
        ]
        return pts, t0.isoformat().replace("+00:00", "Z")  # type: ignore[union-attr]

    # No timestamps: assume uniform 1 s spacing over the track.
    return [TrackPoint(lat, lon, float(i)) for i, (lat, lon, _) in enumerate(raw)], None


def load_route_yaml(path: Path, duration_sec: float) -> list[TrackPoint]:
    """Fallback: a hand-drawn route as `points: [[lat, lon], ...]`.

    Waypoints are spread evenly across the video duration.
    """
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    pts = data.get("points") or []
    if len(pts) < 2:
        raise IngestError(f"{path} needs at least 2 [lat, lon] points")
    span = duration_sec or float(len(pts) - 1)
    step = span / (len(pts) - 1)
    return [TrackPoint(float(p[0]), float(p[1]), i * step) for i, p in enumerate(pts)]


def interpolate(track: list[TrackPoint], t: float) -> tuple[float, float]:
    """Linear interpolation of position at time `t`, clamped to the track."""
    if t <= track[0].t:
        return track[0].lat, track[0].lon
    if t >= track[-1].t:
        return track[-1].lat, track[-1].lon

    lo, hi = 0, len(track) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if track[mid].t <= t:
            lo = mid
        else:
            hi = mid

    a, b = track[lo], track[hi]
    span = b.t - a.t
    f = 0.0 if span <= 0 else (t - a.t) / span
    return a.lat + (b.lat - a.lat) * f, a.lon + (b.lon - a.lon) * f


def smoothed_heading(
    positions: list[tuple[float, float]],
    i: int,
    window: int = 2,
    min_baseline_m: float = 8.0,
) -> float:
    """Compass heading at index `i`, from a baseline long enough to beat GPS noise.

    Heading comes from the direction between two fixes, so its accuracy is
    governed by (travel between them) vs (GPS position noise). Consumer GPS
    carries roughly 3-5 m of error, so the baseline has to be well beyond that.

    Measured on synthetic tracks with 3 m of noise:
        driving 8 m/s, +/-2 samples  ->  median heading error   4 deg
        walking 1.4 m/s, +/-2 samples -> median heading error  38 deg  (p90 107)

    At 1 Hz and walking pace, +/-2 samples spans only ~5.6 m -- less than twice
    the noise, so direction of travel is simply not recoverable from it. Hence
    `min_baseline_m`: keep widening until the two fixes are far enough apart to
    mean something, regardless of how many samples that takes. A 30 deg heading
    error throws a finding 8 m away about 4 m sideways, which is the difference
    between the carriageway and the footpath.
    """
    n = len(positions)
    if n < 2:
        return 0.0

    a = max(0, i - window)
    b = min(n - 1, i + window)
    if a == b:
        a, b = max(0, n - 2), n - 1

    # Widen symmetrically until the baseline is long enough, or we run out.
    while haversine_m(*positions[a], *positions[b]) < min_baseline_m:
        if a == 0 and b == n - 1:
            break  # whole track is shorter than the target baseline
        if a > 0:
            a -= 1
        if b < n - 1:
            b += 1

    return bearing_deg(*positions[a], *positions[b])


def build_frames(
    frame_paths: list[Path],
    frame_times: list[float],
    track: list[TrackPoint],
    frames_subdir: str = "frames",
    time_offset: float = 0.0,
    track_start_iso: str | None = None,
    min_baseline_m: float = 8.0,
) -> list[Frame]:
    """Attach a position and heading to every keyframe.

    `frame_times` are the TRUE source timestamps from extract_frames, not
    reconstructed from an assumed sample rate.

    TWO CLOCKS, kept separate on purpose:

      video time  -- seconds from the first frame. This is what `t_sec` stores,
                     what the timeline scrubs over, and what the video element's
                     currentTime needs.
      GPS time    -- video time plus `time_offset`. Only used to look up where
                     the vehicle was.

    Adding the offset into `t_sec` conflates them, and the symptom is subtle: the
    video panel seeks to t_sec, so with a -2.71 s offset the footage sat 2.71 s
    out of step with its own markers, and the first frame reported t_sec = -2.71
    which the timeline could not even scrub to.
    """
    gps_times = [t + time_offset for t in frame_times]
    positions = [interpolate(track, t) for t in gps_times]
    times = list(frame_times)

    with Image.open(frame_paths[0]) as im:
        width, height = im.size

    base: datetime | None = None
    if track_start_iso:
        try:
            base = datetime.fromisoformat(track_start_iso.replace("Z", "+00:00"))
        except ValueError:
            base = None

    frames: list[Frame] = []
    for i, path in enumerate(frame_paths):
        lat, lon = positions[i]
        ts = None
        if base is not None:
            # `ts` is an absolute wall-clock stamp derived from the GPX start, so
            # it uses GPS time -- unlike t_sec, which stays on the video clock.
            ts = (
                (base + timedelta(seconds=gps_times[i]))
                .isoformat()
                .replace("+00:00", "Z")
            )
        frames.append(
            Frame(
                frame_id=path.stem,
                t_sec=round(times[i], 3),
                ts=ts,
                lat=lat,
                lon=lon,
                heading_deg=round(
                    smoothed_heading(positions, i, min_baseline_m=min_baseline_m), 2
                ),
                path=f"{frames_subdir}/{path.name}",
                width=width,
                height=height,
            )
        )
    return frames


def write_frames(frames: list[Frame], out: Path) -> None:
    out.write_text(json.dumps([f.model_dump() for f in frames], indent=2))


def read_frames(path: Path) -> list[Frame]:
    return [Frame(**d) for d in json.loads(Path(path).read_text())]
