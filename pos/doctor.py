"""Preflight check for your own video + GPX, before spending any API calls.

The failure modes this catches are all silent -- the pipeline runs, produces
findings, and every position is quietly wrong:

  - GPS logger started at a different moment from the recording, so every
    finding is displaced by (offset x speed).
  - Camera left on the shipped default calibration, so every distance is scaled.
  - GPX covering a different time window from the video entirely.
  - Variable frame rate (most phones) -- handled, but worth reporting.

The most useful output is the automatic time-offset suggestion: phone videos
carry a wall-clock `creation_time` and GPX carries absolute timestamps, so the
gap between them can be computed rather than guessed at.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import CameraConfig
from .geo import haversine_m
from .ingest import load_gpx, probe_duration, probe_fps, probe_frame_times


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.warnings: list[str] = []

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  [ok]    {label}" + (f"  --  {detail}" if detail else ""))

    def warn(self, label: str, detail: str = "") -> None:
        print(f"  [WARN]  {label}" + (f"  --  {detail}" if detail else ""))
        self.warnings.append(label)

    def fail(self, label: str, detail: str = "") -> None:
        print(f"  [FAIL]  {label}" + (f"  --  {detail}" if detail else ""))
        self.problems.append(label)


def video_creation_time(video: Path) -> datetime | None:
    """The container's creation_time tag, as a datetime.

    CAREFUL: this is NOT reliably the start of the recording. Many cameras and
    uploaders write it when the file is FINALISED, i.e. the end. Observed on a
    real dashcam clip: tag 10:07:37, duration 43.96 s, GPX 10:06:52-10:07:36 --
    reading the tag as the start gives no overlap with the GPS at all, while
    reading it as the end lines the two up to within a second. run_doctor()
    therefore tests both readings and reports whichever actually overlaps.

    Note also that MP4 stores this at SECOND resolution.
    """
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format_tags=creation_time",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    raw = out.stdout.strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def run_doctor(video: Path, gpx: Path | None, camera: str, fps: float = 2.0) -> int:
    r = Report()

    # ---------------------------------------------------------------- video
    print("\nVIDEO")
    duration = probe_duration(video)
    src_fps = probe_fps(video)

    if duration <= 0:
        r.fail("could not read duration", "is this really a video file?")
    else:
        r.ok("duration", f"{duration:.1f} s")

    if src_fps <= 0:
        r.warn("frame rate unknown", "timestamps will be approximate")
    else:
        r.ok("frame rate", f"{src_fps:.2f} fps nominal")

    times = probe_frame_times(video)
    if times and src_fps > 0:
        gaps = [b - a for a, b in zip(times, times[1:])]
        if gaps:
            nominal = 1.0 / src_fps
            if (max(gaps) - min(gaps)) > nominal * 0.5:
                r.ok(
                    "variable frame rate detected",
                    f"gaps {min(gaps) * 1000:.0f}-{max(gaps) * 1000:.0f} ms; true "
                    "timestamps are read per frame, so this is handled",
                )
            else:
                r.ok("constant frame rate", f"{nominal * 1000:.0f} ms per frame")
    elif not times:
        r.warn("no per-frame timestamps", "falling back to the nominal rate")

    n_keyframes = int(duration * fps) if duration > 0 else 0
    r.ok(f"keyframes at --fps {fps}", f"{n_keyframes} frames -> {n_keyframes} API calls")

    # ------------------------------------------------------------------ gpx
    print("\nGPS TRACK")
    track = None
    start_iso = None
    span = 0.0
    dist = 0.0

    if gpx is None:
        r.warn("no --gpx given", "you will need --route with hand-drawn waypoints")
    elif not gpx.exists():
        r.fail("gpx file not found", str(gpx))
    else:
        try:
            track, start_iso = load_gpx(gpx)
        except Exception as exc:  # noqa: BLE001
            r.fail("gpx could not be parsed", str(exc)[:120])

    if track:
        span = track[-1].t - track[0].t
        dist = sum(
            haversine_m(track[i].lat, track[i].lon, track[i + 1].lat, track[i + 1].lon)
            for i in range(len(track) - 1)
        )
        r.ok("track points", f"{len(track)}")
        r.ok("track duration", f"{span:.1f} s")
        r.ok("track length", f"{dist:.0f} m")

        if span > 0:
            speed = dist / span
            if speed < 0.5:
                r.warn("vehicle barely moved",
                       f"{speed:.1f} m/s -- headings will be unreliable")
            elif speed > 40:
                r.warn("implausible speed", f"{speed:.0f} m/s ({speed * 3.6:.0f} km/h)")
            else:
                r.ok("average speed", f"{speed:.1f} m/s ({speed * 3.6:.0f} km/h)")

        if duration > 0 and 0 < span < duration * 0.5:
            r.fail("track much shorter than the video",
                   f"{span:.0f} s of GPS for {duration:.0f} s of video")

        if not start_iso:
            r.warn("gpx has no timestamps",
                   "points assumed evenly spaced; alignment cannot be checked")

        # --------------------------------------------------- time alignment
        print("\nTIME ALIGNMENT")
        vid_start = video_creation_time(video)
        gpx_start = None
        if start_iso:
            try:
                gpx_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            except ValueError:
                r.warn("could not parse gpx start time", start_iso)

        if vid_start is None:
            r.warn(
                "video has no creation_time metadata",
                "cannot auto-check alignment. If findings sit at the wrong spot, "
                "sweep --time-offset by a few seconds either way.",
            )
        elif gpx_start is None:
            r.warn("gpx has no absolute time", "cannot compare with the video")
        else:
            if vid_start.tzinfo is None:
                vid_start = vid_start.replace(tzinfo=timezone.utc)

            # creation_time may mark the START or the END of the recording --
            # cameras and uploaders disagree. Score both against the GPS window
            # and take whichever actually overlaps.
            gpx_end = gpx_start + timedelta(seconds=span)

            def overlap(v0: datetime) -> float:
                v1 = v0 + timedelta(seconds=duration)
                return (min(v1, gpx_end) - max(v0, gpx_start)).total_seconds()

            as_start = vid_start
            as_end = vid_start - timedelta(seconds=duration)
            ov_s, ov_e = overlap(as_start), overlap(as_end)

            if ov_e > ov_s:
                r.ok(
                    "creation_time is the recording END",
                    f"overlap {ov_e:.0f} s this way vs {ov_s:.0f} s as a start -- "
                    "the tag was written when the file was finalised",
                )
                vid_start = as_end
            else:
                r.ok("creation_time is the recording START", f"overlap {ov_s:.0f} s")

            offset = (vid_start - gpx_start).total_seconds()
            r.ok("video starts", vid_start.isoformat())
            r.ok("gpx starts", gpx_start.isoformat())

            if abs(offset) < 1.0:
                r.ok("clocks agree", f"offset {offset:+.2f} s -- no flag needed")
            elif abs(offset) > 3600:
                r.warn(
                    "offset over an hour",
                    f"{offset:+.0f} s -- probably a timezone problem rather than a "
                    "real delay. Check your GPS app logs UTC.",
                )
            else:
                speed = (dist / span) if span > 0 else 0.0
                err = f" ~{abs(offset) * speed:.0f} m" if speed else ""
                print()
                print(f"  ==> ADD THIS FLAG:  --time-offset {offset:.2f}")
                print(f"      Recording began {offset:+.2f} s relative to the GPS track.")
                print(f"      Without it every finding is displaced by{err}.")
                # The MP4 container stores creation_time at SECOND resolution --
                # ffmpeg truncates a requested .500 to .000 -- so this figure
                # carries up to half a second of residual error. Say so rather
                # than implying precision we do not have.
                if speed:
                    print(
                        f"      NOTE: mp4 creation_time is second-resolution, so this "
                        f"is +/-0.5 s\n            = +/-{0.5 * speed:.1f} m residual at "
                        f"{speed:.1f} m/s. If markers sit consistently ahead of or "
                        f"behind\n            the true spot, nudge it by +/-0.5 and "
                        f"re-run localize (cached, free)."
                    )
                r.warnings.append("time offset needed")

    # --------------------------------------------------------------- camera
    print("\nCAMERA CALIBRATION")
    cam = CameraConfig.load(camera)
    path = Path("configs/camera") / f"{camera}.yaml"
    r.ok("config", str(path) if path.exists() else f"{camera} (built-in defaults)")
    print(f"          height_m={cam.height_m}  vfov_deg={cam.vfov_deg}  "
          f"pitch={cam.pitch_offset_frac:+.4f}")

    default = CameraConfig()
    if (
        abs(cam.height_m - default.height_m) < 1e-9
        and abs(cam.vfov_deg - default.vfov_deg) < 1e-9
        and abs(cam.pitch_offset_frac) < 1e-9
    ):
        r.warn(
            "camera looks uncalibrated",
            "these are the shipped defaults; every distance scales with them. "
            "Run scripts/calibrate.py",
        )
    else:
        r.ok("calibration differs from defaults", "looks measured")

    if cam.pitch_offset_frac == 0.0:
        r.warn(
            "pitch_offset_frac is exactly 0",
            "that assumes the horizon sits exactly on the vertical centre line. "
            "Check one frame -- a tilted dash mount is the usual case.",
        )

    # ------------------------------------------------------------------ api
    print("\nPERCEPTION")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if os.environ.get("NVIDIA_API_KEY", "").strip():
        r.ok("NVIDIA_API_KEY present",
             f"model {os.environ.get('POS_MODEL', '(default)')}")
    else:
        r.warn("no NVIDIA_API_KEY", "only --backend mock will work")

    # --------------------------------------------------------------- verdict
    print("\n" + "=" * 66)
    if r.problems:
        print(f"{len(r.problems)} PROBLEM(S) -- fix before running:")
        for p in r.problems:
            print(f"  - {p}")
        return 1
    if r.warnings:
        print(f"Usable, with {len(r.warnings)} warning(s):")
        for w in r.warnings:
            print(f"  - {w}")
        return 0
    print("All good. Nothing to fix.")
    return 0
