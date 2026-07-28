"""Serve a run directory, plus the SSE stream that drives live mode.

One endpoint covers both the "moving vehicle, alerts arriving as we drive"
demo and a replay of a finished run: `/stream` walks findings in timestamp
order and emits them paced by `speed`. The viewer cannot tell whether findings
are arriving from a live pass or a saved one, which means one code path to keep
working instead of two.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

# File/Form/UploadFile must be imported at MODULE level, not inside create_app.
# `from __future__ import annotations` turns every annotation into a string, and
# FastAPI resolves those against the module namespace -- a function-local import
# leaves UploadFile as an unresolvable ForwardRef and every upload 500s.
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import CameraConfig, DomainConfig, available_domains
from .jobs import (
    MAX_UPLOAD_BYTES,
    TRACK_SUFFIXES,
    VIDEO_SUFFIXES,
    JobStore,
    list_runs,
    safe_run_dir,
)

_FINDING_KEYS = (
    "finding_id",
    "cls",
    "label",
    "lat",
    "lon",
    "severity",
    "confidence",
    "geometry",
)


def create_app(
    run_dir: Path,
    runs_dir: Path | None = None,
    uploads_dir: Path | None = None,
) -> FastAPI:
    """Serve one run, and optionally any run under `runs_dir` plus uploads.

    Every run-scoped endpoint takes an optional `?run=<name>` parameter. Omitted,
    it serves `run_dir` exactly as before, so an existing viewer keeps working.
    Supplied, it serves that run out of `runs_dir` -- which is what lets one
    server show a freshly uploaded job's output without a restart.
    """
    run_dir = Path(run_dir).resolve()
    runs_dir = Path(runs_dir).resolve() if runs_dir else None
    app = FastAPI(title="PhysicalOS", docs_url=None, redoc_url=None)

    jobs: JobStore | None = None
    if runs_dir is not None and uploads_dir is not None:
        jobs = JobStore(Path(uploads_dir), runs_dir)

    # The viewer runs on Vite's dev port during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def _rd(run: str | None) -> Path | None:
        """Which run directory does this request mean?

        Returns None when a name was given but does not resolve to a real run
        inside runs_dir -- safe_run_dir rejects traversal, so "../../etc" cannot
        become a readable path.
        """
        if not run:
            return run_dir
        if runs_dir is None:
            return None
        return safe_run_dir(runs_dir, run)

    def _json_file(rd: Path | None, name: str, fallback):
        if rd is None:
            return JSONResponse({"error": "unknown run"}, status_code=404)
        path = rd / name
        if not path.exists():
            return JSONResponse(fallback)
        return JSONResponse(json.loads(path.read_text()))

    @app.get("/api/manifest")
    def manifest(run: str | None = None):
        rd = _rd(run)
        if rd is None:
            return JSONResponse({"error": f"unknown run: {run}"}, status_code=404)
        path = rd / "manifest.json"
        if not path.exists():
            return JSONResponse({"error": f"no manifest in {rd}"}, status_code=404)
        data = json.loads(path.read_text())
        data["run_dir"] = rd.name

        # Ship the taxonomy with the run so the viewer gets labels, colours and
        # alert flags without duplicating the YAML in TypeScript.
        try:
            dom = DomainConfig.load(data.get("domain", "road"))
            # class_map, not `classes`: it also contains the synthesised absence
            # classes, which the viewer needs in order to label and colour them.
            absence_keys = {a.key for a in dom.absences}
            data["classes"] = [
                {
                    "key": c.key,
                    "label": c.label,
                    "color": c.color,
                    "geometry": c.geometry,
                    "alert": c.alert,
                    "weight": c.weight,
                    # Inferred from coverage rather than detected in a frame, so
                    # the viewer draws no bounding box and marks it as missing.
                    "absence": c.key in absence_keys,
                }
                for c in dom.class_map.values()
            ]
            data["index_name"] = dom.index_name
        except FileNotFoundError:
            data["classes"] = []

        data["has_pointcloud"] = (rd / "cloud.ply").exists()

        # Tell the viewer whether the source clip is actually reachable, so it
        # can hide the video panel rather than showing a broken player.
        try:
            root = Path(__file__).resolve().parent.parent
            vt = (root / data.get("video", "")).resolve()
            data["has_video"] = str(vt).startswith(str(root)) and vt.exists()
        except (OSError, ValueError):
            data["has_video"] = False
        return JSONResponse(data)

    @app.get("/api/findings")
    def findings(run: str | None = None):
        return _json_file(_rd(run), "findings.json", [])

    @app.get("/api/segments")
    def segments(run: str | None = None):
        return _json_file(_rd(run), "segments.json", [])

    @app.get("/api/frames")
    def frames(run: str | None = None):
        return _json_file(_rd(run), "frames.json", [])

    @app.get("/api/twin")
    def twin(run: str | None = None):
        return _json_file(_rd(run), "twin.json", {"buildings": [], "roads": []})

    @app.get("/api/coverage")
    def coverage(run: str | None = None):
        """Per-asset coverage: how many were found, and how many gaps."""
        return _json_file(_rd(run), "coverage.json", [])

    @app.get("/api/review.mp4")
    def review_video(run: str | None = None):
        """Serve the annotated review render, if `pos video` has been run.

        Separate from /api/video, which streams the SOURCE clip: this is the
        derived artifact and lives inside the run directory, so it needs none of
        that endpoint's escape checking -- there is no manifest-supplied path
        here to distrust. 404 rather than an error page when absent, because not
        every run has one and the caller decides whether to offer the link.
        """
        rd = _rd(run)
        if rd is None:
            return JSONResponse({"error": "unknown run"}, status_code=404)
        target = rd / "review.mp4"
        if not target.exists():
            return JSONResponse({"error": "no review video for this run"}, status_code=404)
        return FileResponse(target, media_type="video/mp4")

    @app.get("/api/video")
    def video(request: Request, run: str | None = None):
        """Stream the source video, with Range support.

        The <video> element needs byte-range requests to seek; without them a
        73 MB clip would have to download in full before the first frame shows.
        The path comes from manifest.video and is resolved relative to the repo
        root -- and rejected if it escapes it, so a hand-edited manifest cannot
        be used to read arbitrary files off disk.
        """
        rd = _rd(run)
        if rd is None:
            return JSONResponse({"error": "unknown run"}, status_code=404)
        mpath = rd / "manifest.json"
        if not mpath.exists():
            return JSONResponse({"error": "no manifest"}, status_code=404)

        rel = json.loads(mpath.read_text()).get("video", "")
        root = Path(__file__).resolve().parent.parent
        target = (root / rel).resolve()
        if not str(target).startswith(str(root)) or not target.exists():
            return JSONResponse(
                {"error": f"video not available: {rel}"}, status_code=404
            )

        size = target.stat().st_size
        rng = request.headers.get("range")
        if not rng:
            return FileResponse(target, media_type="video/mp4")

        # "bytes=START-END" -- END optional.
        try:
            spec = rng.split("=", 1)[1]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except (IndexError, ValueError):
            return FileResponse(target, media_type="video/mp4")

        start = max(0, min(start, size - 1))
        end = max(start, min(end, size - 1))
        length = end - start + 1

        def chunks(chunk_size: int = 1 << 18):
            with open(target, "rb") as fh:
                fh.seek(start)
                left = length
                while left > 0:
                    data = fh.read(min(chunk_size, left))
                    if not data:
                        break
                    left -= len(data)
                    yield data

        return StreamingResponse(
            chunks(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            },
        )

    @app.get("/api/basemap")
    def basemap(run: str | None = None):
        """Bounds and attribution for the cached satellite texture.

        Returns `{"available": false}` rather than a 404 when absent: the
        basemap is optional and the viewer must be able to ask without treating
        the answer as an error.
        """
        rd = _rd(run)
        if rd is None:
            return JSONResponse({"available": False})
        path = rd / "basemap.json"
        if not path.exists() or not (rd / "basemap.jpg").exists():
            return JSONResponse({"available": False})
        data = json.loads(path.read_text())
        data["available"] = True
        return JSONResponse(data)

    @app.get("/api/basemap.jpg")
    def basemap_image(run: str | None = None):
        rd = _rd(run)
        path = (rd / "basemap.jpg") if rd else None
        if path is None or not path.exists():
            return JSONResponse(
                {"error": "no basemap -- run `pos basemap`"}, status_code=404
            )
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/kml")
    def kml(run: str | None = None):
        """The run as a KMZ for Google Earth, built on demand and cached.

        Rebuilt whenever findings.json is newer than the cached archive, so a
        re-cluster or re-score is picked up without anyone having to remember to
        regenerate it.
        """
        from .kmlexport import build_kmz

        rd = _rd(run)
        if rd is None:
            return JSONResponse({"error": "unknown run"}, status_code=404)
        out = rd / "export.kmz"
        src = rd / "findings.json"
        if not src.exists():
            return JSONResponse({"error": "no findings in this run"}, status_code=404)

        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            try:
                build_kmz(rd, out)
            except (OSError, KeyError, ValueError) as exc:
                return JSONResponse(
                    {"error": f"kmz build failed: {exc}"}, status_code=500
                )

        run_id = "run"
        try:
            run_id = json.loads((rd / "manifest.json").read_text()).get("run_id", "run")
        except (OSError, ValueError):
            pass

        return FileResponse(
            out,
            media_type="application/vnd.google-earth.kmz",
            filename=f"physicalos-{run_id}.kmz",
        )

    @app.get("/api/report.pdf")
    def report_pdf(download: bool = False, run: str | None = None):
        """The PDF inspection report, built on demand and cached.

        Same staleness rule as /api/kml: rebuilt whenever findings.json is newer,
        so a re-cluster, a re-score or a verify pass is picked up without anyone
        remembering to regenerate it.

        Served INLINE by default so the browser's own PDF viewer opens it in a
        tab; `?download=1` forces a save dialog instead. Content-Disposition is
        the only thing that decides which happens.
        """
        from .report import build_report

        rd = _rd(run)
        if rd is None:
            return JSONResponse({"error": "unknown run"}, status_code=404)
        out = rd / "report.pdf"
        src = rd / "findings.json"
        if not src.exists():
            return JSONResponse({"error": "no findings in this run"}, status_code=404)

        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            try:
                build_report(rd, out)
            except (OSError, KeyError, ValueError) as exc:
                return JSONResponse(
                    {"error": f"pdf build failed: {exc}"}, status_code=500
                )

        run_id = "run"
        try:
            run_id = json.loads((rd / "manifest.json").read_text()).get("run_id", "run")
        except (OSError, ValueError):
            pass

        name = f"physicalos-{run_id}.pdf"
        disposition = "attachment" if download else "inline"
        return FileResponse(
            out,
            media_type="application/pdf",
            headers={"Content-Disposition": f'{disposition}; filename="{name}"'},
        )

    @app.get("/api/cloud.ply")
    def cloud(run: str | None = None):
        rd = _rd(run)
        path = (rd / "cloud.ply") if rd else None
        if path is None or not path.exists():
            return JSONResponse(
                {"error": "no point cloud in this run"}, status_code=404
            )
        return FileResponse(path, media_type="application/octet-stream")

    @app.get("/stream")
    async def stream(speed: float = 1.0, run: str | None = None):
        """Replay findings in time order as Server-Sent Events.

        `speed` multiplies real time. The default is 1.0 -- real time -- because
        the viewer plays the source video alongside at its natural rate, and any
        other pacing would leave the footage drifting behind the markers. Pass a
        higher value only if you are consuming the stream without the video.
        """
        rd = _rd(run) or run_dir
        path = rd / "findings.json"
        items = json.loads(path.read_text()) if path.exists() else []
        items.sort(key=lambda f: f.get("t_sec", 0.0))

        alert_classes: set[str] = set()
        try:
            dom_key = json.loads((rd / "manifest.json").read_text()).get(
                "domain", "road"
            )
            alert_classes = {
                c.key for c in DomainConfig.load(dom_key).classes if c.alert
            }
        except Exception:  # noqa: BLE001 - alerts are cosmetic, never fatal
            pass

        async def gen():
            previous = 0.0
            for item in items:
                t = float(item.get("t_sec", 0.0))
                delay = max(0.0, (t - previous) / max(speed, 0.1))
                # Cap the wait so a quiet stretch does not stall the demo.
                await asyncio.sleep(min(delay, 2.0))
                previous = t

                payload = {
                    "type": "finding",
                    "t_sec": t,
                    "finding": {
                        **{k: item.get(k) for k in _FINDING_KEYS},
                        "alert": item.get("cls") in alert_classes,
                        "n_evidence": len(item.get("evidence") or []),
                    },
                }
                yield f"data: {json.dumps(payload)}\n\n"

            yield f"data: {json.dumps({'type': 'done', 't_sec': previous})}\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------------------------------------------------------- evidence images
    # A route rather than a StaticFiles mount, because the image has to come from
    # whichever run the request names.
    @app.get("/frames/{name}")
    def frame_image(name: str, run: str | None = None):
        rd = _rd(run)
        # Only a bare filename: no separators, so nothing can climb out of frames/.
        if rd is None or "/" in name or "\\" in name or name.startswith("."):
            return JSONResponse({"error": "not found"}, status_code=404)
        path = rd / "frames" / name
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    # ------------------------------------------------------------ runs + upload
    @app.get("/api/runs")
    def api_runs():
        """Every processed run available to view."""
        if runs_dir is None:
            return JSONResponse(
                [{"run": run_dir.name, "single_run_mode": True}]
            )
        return JSONResponse(list_runs(runs_dir))

    if jobs is not None:

        @app.post("/api/upload")
        async def upload(
            video: UploadFile = File(...),
            gpx: UploadFile = File(...),
            camera: str = Form("dashcam"),
            # "auto" solves calibration from this clip. These two only matter in
            # that case: the assumed mounting height, and what to fall back to
            # when the fit comes out unusable.
            camera_height: float = Form(1.2),
            fallback_camera: str = Form("dashcam"),
            domain: str = Form("road_pci"),
            backend: str = Form("ensemble"),
            spacing_m: float = Form(2.0),
            tile: int = Form(0),
            name: str = Form(""),
        ):
            """Accept a video + GPX and start a pipeline job.

            The extension allow-list is the only file validation worth doing here:
            the bytes go to ffmpeg and gpxpy, which reject nonsense themselves, and
            nothing is ever executed. What matters is that the stored NAME cannot
            escape the uploads directory, so only the suffix is trusted and the
            stem is discarded.
            """
            v_suffix = Path(video.filename or "").suffix.lower()
            g_suffix = Path(gpx.filename or "").suffix.lower()
            if v_suffix not in VIDEO_SUFFIXES:
                return JSONResponse(
                    {"error": f"video must be one of {sorted(VIDEO_SUFFIXES)}"},
                    status_code=400,
                )
            if g_suffix not in TRACK_SUFFIXES:
                return JSONResponse(
                    {"error": f"track must be one of {sorted(TRACK_SUFFIXES)}"},
                    status_code=400,
                )

            run_id = re.sub(r"[^a-zA-Z0-9_-]", "", name).strip("-_") or (
                "upload-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            )
            if (runs_dir / run_id / "manifest.json").exists():
                run_id = f"{run_id}-{uuid.uuid4().hex[:4]}"

            stage = Path(jobs.uploads_dir) / run_id
            stage.mkdir(parents=True, exist_ok=True)
            vpath = stage / f"video{v_suffix}"
            gpath = stage / f"track{g_suffix}"

            # Stream to disk in chunks: a 350 MB clip must not be held in memory.
            total = 0
            for src, dest in ((video, vpath), (gpx, gpath)):
                with open(dest, "wb") as fh:
                    while chunk := await src.read(1 << 20):
                        total += len(chunk)
                        if total > MAX_UPLOAD_BYTES:
                            fh.close()
                            return JSONResponse(
                                {"error": "upload exceeds 2 GB"}, status_code=413
                            )
                        fh.write(chunk)

            job = jobs.create(
                run_id=run_id,
                video=vpath,
                gpx=gpath,
                camera=camera,
                domain=domain,
                backend=backend,
                spacing_m=spacing_m,
                tile=tile,
                model_path=os.environ.get("POS_ONNX"),
                camera_height=camera_height,
                fallback_camera=fallback_camera,
            )
            return JSONResponse(job.as_dict(), status_code=202)

        @app.get("/api/jobs")
        def api_jobs():
            return JSONResponse(jobs.list())

        @app.get("/api/jobs/{job_id}")
        def api_job(job_id: str):
            job = jobs.get(job_id)
            if job is None:
                return JSONResponse({"error": "unknown job"}, status_code=404)
            return JSONResponse(job.as_dict())

        @app.get("/api/jobs/{job_id}/events")
        async def api_job_events(job_id: str):
            """Progress as Server-Sent Events, so the page needs no polling loop."""

            async def gen():
                sent = 0
                while True:
                    job = jobs.get(job_id)
                    if job is None:
                        yield f"data: {json.dumps({'type': 'error', 'error': 'unknown job'})}\n\n"
                        return
                    lines = job.log[sent:]
                    if lines:
                        sent = len(job.log)
                        yield "data: " + json.dumps(
                            {"type": "log", "lines": lines, "stage": job.stage}
                        ) + "\n\n"
                    if job.status in ("done", "failed"):
                        yield "data: " + json.dumps(
                            {
                                "type": job.status,
                                "run": job.run_id,
                                "error": job.error,
                                "time_offset": job.time_offset,
                            }
                        ) + "\n\n"
                        return
                    await asyncio.sleep(1.0)

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @app.get("/api/cameras")
        def api_cameras():
            """Camera configs and domains the upload form can offer.

            Calibration is the thing that most needs getting right: the shipped
            default vertical FOV is 58 degrees where a real clip measured 24.74,
            a 2x range error that produces output looking entirely plausible.
            The `auto` entry is not a config file -- it tells the job to solve
            the geometry from the uploaded clip, and to fall back to a preset if
            the fit comes out unusable.
            """
            # .env is loaded lazily elsewhere (doctor, cosmos), so without this
            # the form would report "no API key" while the job subprocess -- which
            # does load it -- would happily use the VLM. Reporting the wrong
            # capability pushes the user to a weaker backend for no reason.
            try:
                from dotenv import load_dotenv

                load_dotenv(Path(__file__).resolve().parent.parent / ".env")
            except ImportError:
                pass

            cams = [
                {
                    "name": "auto",
                    "height_m": None,
                    "vfov_deg": None,
                    "pitch_offset_frac": None,
                    "is_default": False,
                    "solves_per_video": True,
                }
            ]
            cdir = Path(__file__).resolve().parent.parent / "configs" / "camera"
            for f in sorted(cdir.glob("*.yaml")):
                try:
                    cam = CameraConfig.load(f.stem)
                    cams.append(
                        {
                            "name": f.stem,
                            "height_m": cam.height_m,
                            "vfov_deg": cam.vfov_deg,
                            "pitch_offset_frac": cam.pitch_offset_frac,
                            "is_default": f.stem == "dashcam",
                        }
                    )
                except Exception:  # noqa: BLE001
                    continue
            data = {
                "cameras": cams,
                "domains": available_domains(),
                "has_api_key": bool(os.environ.get("NVIDIA_API_KEY")),
            }
            try:
                from .perception.onnx_yolo import resolve_model_path
                data["has_onnx"] = bool(resolve_model_path(None))
            except Exception:
                data["has_onnx"] = False

            return JSONResponse(data)

        # The upload page. Served explicitly so it wins over the viewer mount.
        @app.get("/studio")
        def studio():
            page = Path(__file__).resolve().parent / "studio.html"
            if not page.exists():
                return JSONResponse({"error": "studio.html missing"}, status_code=500)
            return FileResponse(page, media_type="text/html")

    # The built viewer, when it exists. Mounted last so /api and /stream win.
    dist = Path(__file__).resolve().parent.parent / "viewer" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="viewer")

    return app


def serve(
    run_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    runs_dir: Path | None = None,
    uploads_dir: Path | None = None,
) -> None:
    import uvicorn

    app = create_app(run_dir, runs_dir=runs_dir, uploads_dir=uploads_dir)
    dist = Path(__file__).resolve().parent.parent / "viewer" / "dist"

    print(f"\n  PhysicalOS serving {Path(run_dir).resolve()}")
    print(f"  API      http://{host}:{port}/api/manifest")
    print(f"  Stream   http://{host}:{port}/stream?speed=1   (1 = real time)")
    if dist.exists():
        print(f"  Viewer   http://{host}:{port}/")
    else:
        print("  Viewer   not built -- run `npm install && npm run dev` in viewer/")
    if runs_dir is not None and uploads_dir is not None:
        print(f"  STUDIO   http://{host}:{port}/studio   <- upload a video + GPX here")
        print(f"           runs in {Path(runs_dir).resolve()}")
    print()

    uvicorn.run(app, host=host, port=port, log_level="warning")
