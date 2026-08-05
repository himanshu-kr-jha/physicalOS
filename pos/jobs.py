"""Background pipeline jobs, so a video + GPX can be processed from the browser.

WHY A SUBPROCESS AND NOT A THREAD
The pipeline shells out to ffmpeg, loads onnxruntime, and runs for minutes. In a
thread it would block the event loop that serves the progress stream, and an
onnxruntime crash would take the web server down with it. A subprocess isolates
both, and its stdout is already a perfectly good progress feed -- the CLI prints
"=== perceive ===" and "40/102 frames" as it goes, so no extra instrumentation
is needed.

WHY doctor RUNS FIRST, AUTOMATICALLY
This is the one piece of judgement a web form cannot ask a user for. A video's
`creation_time` is frequently the moment the file was FINALISED rather than when
recording began, so the naive reading puts the GPS track out of step with the
footage -- measured on real footage at -2.71 s, about 6 m of displacement on every
finding. Nothing about the output looks wrong when it happens. So the job runs
`pos doctor`, parses the offset it recommends, and passes that to the pipeline
instead of trusting a default of zero.

CALIBRATION, AND WHY IT IS ALLOWED TO REFUSE
An uncalibrated run produces findings that are all plausible and all wrong: the
shipped default vertical FOV is 58 degrees where one real clip measured 24.74, a
2x range error, and nothing about the output looks suspect. So the job can now
solve calibration from the clip itself -- camera "auto" runs
scripts/calibrate_from_motion.py against the uploaded video and GPX and uses the
answer for THAT RUN ONLY: written beside the upload, never into configs/, never
shared with another run.

The important half is that it validates the answer and refuses a bad one. That
solver is unreliable at low speed by construction: with little forward motion
only near features move, they all land in a narrow band of rows, and the horizon
becomes unidentifiable -- a real bike clip in this repo fits to a 2.5 degree
vertical FOV, which is nonsense. The solver reports those conditions as warnings
in a JSON sidecar; on any warning, or a failed fit, the job falls back to the
preset the user picked and says so in the log. Guessing silently would be worse
than not guessing at all.

THE CSV PATH, AND WHY IT SKIPS HALF THE PIPELINE
An upload may carry a CSV of findings someone else produced -- a field team's
spreadsheet, or a run exported with `pos export-csv` and edited. Those rows already
carry a lat/lon, so there is nothing to detect and nothing to project: perceive,
localize and cluster have no work to do, and calibration is meaningless because no
pixel is ever turned into a distance. That path runs ingest (for the keyframes, the
route and the video), then import-csv, score and twin, and finishes in seconds
without needing a detector model or an API key.

`twin` is invoked EXPLICITLY there. On the detector path it comes free inside
`pos run`; forgetting it here is the difference between a viewer with extruded OSM
buildings and one with a bare grey ground plane.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Upload ceiling. A 200 MB phone clip is normal; 4 GB is someone's mistake.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
TRACK_SUFFIXES = {".gpx", ".xml"}
#: A findings list someone else produced. Columns are whatever `pos export-csv`
#: writes, so a scored run round-trips out to a spreadsheet and back.
CSV_SUFFIXES = {".csv"}

# `pos doctor` prints:  ==> ADD THIS FLAG:  --time-offset -2.71
_OFFSET_RE = re.compile(r"--time-offset\s+(-?\d+(?:\.\d+)?)")

# Camera value meaning "solve it from this clip" rather than naming a preset.
AUTO_CAMERA = "auto"

# Two shots at the motion fit before giving up. The first is the tuned road
# window from the script's own docs. The second widens the band and lowers the
# speed floor, which finds far more features on slower or bumpier footage --
# more features are not automatically a better fit, so both are validated the
# same way and the first CLEAN one wins.
_CAL_ATTEMPTS: tuple[tuple[str, list[str]], ...] = (
    ("road ROI", ["--pairs", "16", "--roi-top", "0.42", "--roi-bottom", "0.66"]),
    (
        "wide ROI, lower speed floor",
        ["--pairs", "24", "--dt", "0.5", "--min-speed", "2.0",
         "--roi-top", "0.38", "--roi-bottom", "0.78"],
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Job:
    job_id: str
    run_id: str
    status: str = "queued"          # queued | running | done | failed
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    stage: str = ""
    log: list[str] = field(default_factory=list)
    error: str | None = None
    returncode: int | None = None
    time_offset: float | None = None
    #: The solved calibration, or the record of why it was rejected. Surfaced so
    #: the studio can show which geometry the numbers on screen actually rest on.
    calibration: dict | None = None
    args: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "status": self.status,
            "created": self.created,
            "updated": self.updated,
            "stage": self.stage,
            "error": self.error,
            "returncode": self.returncode,
            "time_offset": self.time_offset,
            "calibration": self.calibration,
            "args": self.args,
            # Tail only: a 600-frame run emits hundreds of progress lines and the
            # browser only ever displays the recent ones.
            "log": self.log[-200:],
            "n_log": len(self.log),
        }


class JobStore:
    """In-memory jobs, mirrored to disk so a restart does not lose history."""

    def __init__(self, uploads_dir: Path, runs_dir: Path):
        self.uploads_dir = Path(uploads_dir)
        self.runs_dir = Path(runs_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load_existing()

    # ----------------------------------------------------------------- disk

    def _job_file(self, job_id: str) -> Path:
        return self.uploads_dir / job_id / "job.json"

    def _load_existing(self) -> None:
        for jf in sorted(self.uploads_dir.glob("*/job.json")):
            try:
                d = json.loads(jf.read_text())
                job = Job(
                    job_id=d["job_id"],
                    run_id=d.get("run_id", ""),
                    status=d.get("status", "failed"),
                    created=d.get("created", _now()),
                    updated=d.get("updated", _now()),
                    stage=d.get("stage", ""),
                    log=d.get("log", []),
                    error=d.get("error"),
                    returncode=d.get("returncode"),
                    time_offset=d.get("time_offset"),
                    calibration=d.get("calibration"),
                    args=d.get("args", {}),
                )
                # A job recorded as running cannot still be running after a
                # restart: its subprocess died with the previous server.
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.error = "server restarted while this job was running"
                self._jobs[job.job_id] = job
            except (OSError, ValueError, KeyError):
                continue

    def _persist(self, job: Job) -> None:
        try:
            f = self._job_file(job.job_id)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(job.as_dict(), indent=2))
        except OSError:
            pass  # progress reporting must never break the run itself

    # ------------------------------------------------------------------ api

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
            return [j.as_dict() for j in jobs]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _touch(self, job: Job, stage: str | None = None) -> None:
        job.updated = _now()
        if stage:
            job.stage = stage
        self._persist(job)

    def create(
        self,
        run_id: str,
        video: Path,
        gpx: Path,
        camera: str,
        domain: str,
        backend: str,
        spacing_m: float,
        tile: int,
        model_path: str | None = None,
        camera_height: float = 1.2,
        fallback_camera: str = "dashcam",
        csv: Path | None = None,
    ) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            run_id=run_id,
            args={
                "camera": camera,
                "camera_height": camera_height,
                "fallback_camera": fallback_camera,
                "domain": domain,
                # Recorded as-supplied, but see _run: on the CSV path no detector
                # runs, so this value is not what produced the findings.
                "backend": backend,
                "spacing_m": spacing_m,
                "tile": tile,
                "video": video.name,
                "gpx": gpx.name,
                "csv": csv.name if csv else None,
            },
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._persist(job)

        threading.Thread(
            target=self._run,
            args=(job, video, gpx, camera, domain, backend, spacing_m, tile,
                  model_path, camera_height, fallback_camera, csv),
            daemon=True,
        ).start()
        return job

    # --------------------------------------------------------------- runner

    def _stream(self, job: Job, cmd: list[str], stage: str) -> int:
        """Run one command, appending its output to the job log line by line."""
        self._touch(job, stage)
        # Two command shapes reach here: `python -m pos.cli <sub> ...` and
        # `python scripts/<name>.py ...`. Slicing a fixed offset off the front
        # mangles the second, which then reads as if the first flag were missing.
        if len(cmd) > 2 and cmd[1] == "-m":
            job.log.append(f"$ pos {' '.join(cmd[3:])}")
        else:
            job.log.append(f"$ {Path(cmd[1]).name} {' '.join(cmd[2:])}")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            job.log.append(f"could not start: {exc}")
            return 1

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                job.log.append(line)
                # Persist periodically, not per line: a 600-frame run would
                # otherwise rewrite job.json thousands of times.
                if len(job.log) % 20 == 0:
                    self._persist(job)
        proc.wait()
        self._persist(job)
        return proc.returncode or 0

    def _calibrate(
        self,
        job: Job,
        video: Path,
        gpx: Path,
        offset: float,
        height: float,
        fallback: str,
    ) -> str:
        """Solve calibration from this clip. Returns the --camera to actually use.

        Falls back to `fallback` on any failure or warning, loudly. The caller
        gets a plain string either way, so the pipeline command is unaffected by
        which path was taken.
        """
        work = self.uploads_dir / job.job_id
        script = REPO_ROOT / "scripts" / "calibrate_from_motion.py"

        for n, (label, extra) in enumerate(_CAL_ATTEMPTS, 1):
            yaml_path = work / f"camera_auto{n}.yaml"
            json_path = work / f"calibration{n}.json"
            job.log.append(f"calibration attempt {n}/{len(_CAL_ATTEMPTS)}: {label}")

            self._stream(
                job,
                [sys.executable, str(script),
                 "--video", str(video), "--gpx", str(gpx),
                 "--height", str(height),
                 "--time-offset", str(offset),
                 "--write", str(yaml_path),
                 "--json", str(json_path),
                 *extra],
                "calibration",
            )

            try:
                result = json.loads(json_path.read_text())
            except (OSError, ValueError):
                job.log.append(f"  attempt {n}: no result written")
                continue

            if not result.get("ok"):
                job.log.append(f"  attempt {n}: {result.get('error')}")
                continue
            if result.get("warnings"):
                for w in result["warnings"]:
                    job.log.append(f"  attempt {n} REJECTED: {w}")
                continue

            job.calibration = {**result, "source": "auto", "attempt": label}
            job.log.append(
                f"  calibrated from this clip: vfov {result['vfov_deg']}deg, "
                f"horizon row {result['horizon_row']} of {result['frame'][1]}, "
                f"residual {result['residual_m']} m over {result['inliers']} "
                f"tracked features"
            )
            self._persist(job)
            return str(yaml_path)

        job.calibration = {
            "ok": False,
            "source": "preset",
            "preset": fallback,
            "error": "auto-calibration produced no usable fit",
        }
        job.log.append(
            f"AUTO-CALIBRATION FAILED -- falling back to preset '{fallback}'. "
            f"Ranges are only as good as that preset; if it does not match this "
            f"camera every distance is wrong in proportion."
        )
        self._persist(job)
        return fallback

    def _run(
        self,
        job: Job,
        video: Path,
        gpx: Path,
        camera: str,
        domain: str,
        backend: str,
        spacing_m: float,
        tile: int,
        model_path: str | None,
        camera_height: float = 1.2,
        fallback_camera: str = "dashcam",
        csv: Path | None = None,
    ) -> None:
        py = sys.executable
        out = self.runs_dir / job.run_id
        job.status = "running"
        self._touch(job, "preflight")

        try:
            # `doctor` needs a camera to sanity-check against, and "auto" is not
            # one yet -- calibration runs after it, because the fit wants the
            # clock offset doctor is about to recover.
            preflight_camera = fallback_camera if camera == AUTO_CAMERA else camera

            # 1. Preflight, to recover the clock offset. Its exit code is
            # non-zero on warnings too, so it is never treated as fatal.
            self._stream(
                job,
                [py, "-m", "pos.cli", "doctor", "--video", str(video),
                 "--gpx", str(gpx), "--camera", preflight_camera],
                "preflight",
            )
            offset = 0.0
            for line in job.log:
                m = _OFFSET_RE.search(line)
                if m:
                    offset = float(m.group(1))
            job.time_offset = offset
            job.log.append(
                f"using --time-offset {offset}" + ("" if offset else "  (no shift needed)")
            )
            self._persist(job)

            # 2. Calibration, solved from this very clip when asked for. It uses
            # the offset above, so it has to come after preflight and before the
            # pipeline -- every range in the run depends on its answer.
            #
            # Not on the CSV path: calibration exists to turn a pixel row into a
            # distance, and no pixel is measured there. Solving it anyway would burn
            # a minute to produce a number nothing reads.
            if camera == AUTO_CAMERA:
                if csv is not None:
                    camera = fallback_camera
                    job.log.append(
                        f"calibration skipped: the CSV supplies each finding's "
                        f"position, so no pixel is projected. Using preset "
                        f"'{camera}' for the render stages that need a camera."
                    )
                else:
                    camera = self._calibrate(
                        job, video, gpx, offset, camera_height, fallback_camera
                    )
                job.args["resolved_camera"] = camera
                self._persist(job)

            # 3. The pipeline proper -- or, with a CSV, the short path that skips
            # detection entirely. `spacing_m > 0` selects distance-based keyframing,
            # anything else falls back to one frame a second.
            sampling = (
                ["--spacing-m", str(spacing_m)]
                if spacing_m and spacing_m > 0
                else ["--fps", "1"]
            )

            if csv is not None:
                # Order matters. ingest writes frames.json and the manifest (including
                # manifest.video, which is what turns the viewer's video panel and
                # chase camera on); import-csv then snaps each row to the nearest
                # keyframe in time; score needs both to exist; twin needs the route.
                stages: tuple[tuple[str, list[str], bool], ...] = (
                    ("ingest", [
                        "ingest", "--video", str(video), "--gpx", str(gpx),
                        "--out", str(out), "--domain", domain,
                        "--time-offset", str(offset), "--heading-baseline", "15",
                        *sampling,
                    ], True),
                    ("csv import", [
                        "import-csv", str(csv), "--run", str(out),
                    ], True),
                    # Non-fatal from here: the findings are already viewable, and
                    # these add the heatmap and the buildings on top.
                    ("score", ["score", "--run", str(out)], False),
                    ("osm twin", ["twin", "--run", str(out)], False),
                )
                for stage, extra, fatal in stages:
                    rc = self._stream(job, [py, "-m", "pos.cli", *extra], stage)
                    if rc != 0 and fatal:
                        job.status = "failed"
                        job.error = f"{stage} exited {rc}"
                        job.returncode = rc
                        self._touch(job, "failed")
                        return
            else:
                cmd = [
                    py, "-m", "pos.cli", "run",
                    "--video", str(video), "--gpx", str(gpx),
                    "--out", str(out),
                    "--camera", camera, "--domain", domain, "--backend", backend,
                    "--time-offset", str(offset),
                    "--heading-baseline", "15",
                    *sampling,
                ]
                if tile:
                    cmd += ["--tile", str(tile)]
                if model_path:
                    cmd += ["--model-path", model_path]

                rc = self._stream(job, cmd, "pipeline")
                if rc != 0:
                    job.status = "failed"
                    job.error = f"pipeline exited {rc}"
                    job.returncode = rc
                    self._touch(job, "failed")
                    return

            # Keep the geometry the numbers were produced under beside them. A
            # run whose calibration lived only in uploads/ cannot be re-checked
            # once that upload is cleared out.
            if job.calibration:
                try:
                    (out / "calibration.json").write_text(
                        json.dumps(job.calibration, indent=2) + "\n"
                    )
                    src = Path(camera)
                    if src.suffix == ".yaml" and src.exists():
                        (out / "camera.yaml").write_text(src.read_text())
                except OSError:
                    pass  # provenance is nice to have, never worth failing a run

            # 4. Extras. None may fail the job -- the run is already viewable.
            #    Segmentation comes before the video because the video draws its
            #    mask, and both come after the point cloud only because the
            #    findings are what someone is waiting for, not the render.
            for stage, extra in (
                ("road segmentation", ["segment", "--run", str(out), "--camera", camera]),
                ("review video", ["video", "--run", str(out),
                                  "--out", str(out / "review.mp4")]),
                ("point cloud", ["depthcloud", "--run", str(out), "--camera", camera,
                                 "--stride", "5"]),
                ("satellite basemap", ["basemap", "--run", str(out), "--zoom", "18"]),
                ("google earth export", ["kml", "--run", str(out),
                                         "--out", str(out / "export.kmz")]),
                ("pdf report", ["report", "--run", str(out),
                                "--out", str(out / "report.pdf"),
                                "--image-width", "700"]),
            ):
                self._stream(job, [py, "-m", "pos.cli", *extra], stage)

            job.status = "done"
            job.returncode = 0
            self._touch(job, "done")
        except Exception as exc:  # noqa: BLE001 - a job must never kill the server
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            self._touch(job, "failed")


def safe_run_dir(runs_dir: Path, name: str) -> Path | None:
    """Resolve a run name inside runs_dir, or None if it escapes.

    The name arrives from a query string, so "../../etc" must not resolve to
    anything readable. Resolving both sides and testing containment is the only
    reliable check -- a string prefix test misses symlinks.
    """
    if not name or "\x00" in name:
        return None
    runs_dir = Path(runs_dir).resolve()
    try:
        target = (runs_dir / name).resolve()
    except (OSError, RuntimeError):
        return None
    if target == runs_dir or runs_dir not in target.parents:
        return None
    return target if (target / "manifest.json").exists() else None


def list_runs(runs_dir: Path) -> list[dict]:
    """Every run in runs_dir that has a manifest, newest first."""
    out: list[dict] = []
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return out
    for mf in runs_dir.glob("*/manifest.json"):
        try:
            m = json.loads(mf.read_text())
        except (OSError, ValueError):
            continue
        s = m.get("summary") or {}
        out.append(
            {
                "run": mf.parent.name,
                "domain": m.get("domain"),
                "created": m.get("created"),
                "backend": m.get("backend"),
                "n_findings": m.get("n_findings", 0),
                "n_frames": m.get("n_frames", 0),
                "quality_index": s.get("quality_index"),
                "grade": s.get("grade"),
                "route_m": s.get("route_length_m"),
                "has_pointcloud": (mf.parent / "cloud.ply").exists(),
                "has_basemap": (mf.parent / "basemap.jpg").exists(),
            }
        )
    out.sort(key=lambda r: str(r.get("created") or ""), reverse=True)
    return out
