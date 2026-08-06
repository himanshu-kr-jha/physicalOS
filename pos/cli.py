"""PhysicalOS command line.

Each pipeline stage is its own command so you can re-run one step without
paying for the others -- re-clustering after a radius tweak should never
re-hit a paid API. `pos run` chains them all.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from .cluster import (
    cluster_detections,
    localize_detections,
    read_detections,
    read_findings,
    write_detections,
    write_findings,
)
from .config import CameraConfig, DomainConfig, available_domains
from .ingest import (
    build_frames,
    extract_frames,
    load_gpx,
    load_route_yaml,
    probe_duration,
    read_frames,
    write_frames,
)
from .schema import RunManifest
from .score import build_segments, summarize, write_segments

app = typer.Typer(
    add_completion=False,
    help="PhysicalOS - spatial intelligence from ordinary video.",
    no_args_is_help=True,
)


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _echo(msg: str) -> None:
    typer.echo(msg)


def _die(msg: str) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _detectable_assets(backend: str, dom) -> set[str]:
    """Asset classes the given backend could actually detect.

    The ONNX model knows only its own distress classes -- currently the two
    crack types -- and has no streetlight or footpath output at all. Running
    absence rules against it would mark every route unlit, so those rules are
    skipped unless a VLM is in the loop. Derived from CLASS_KEYS rather than
    listed here, so swapping the weights needs no change in this function.
    """
    from .perception.onnx_yolo import CLASS_KEYS

    all_assets = {r.asset for r in dom.absences}
    if backend == "onnx":
        return all_assets & set(CLASS_KEYS)
    # mock replays fixtures, cosmos/ensemble include a VLM: all assets in play.
    return all_assets


def _load_manifest(run_dir: Path) -> RunManifest:
    path = run_dir / "manifest.json"
    if not path.exists():
        _die(f"No manifest at {path}. Run `pos ingest` first.")
    return RunManifest(**json.loads(path.read_text()))


def _save_manifest(m: RunManifest, run_dir: Path) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(m.model_dump(), indent=2))


# --------------------------------------------------------------------------


@app.command()
def domains() -> None:
    """List the available inspection domains."""
    for key in available_domains():
        d = DomainConfig.load(key)
        _echo(f"{key:22s} {d.label}  ({len(d.classes)} classes)")


@app.command()
def doctor(
    video: Path = typer.Option(..., "--video", "-v", exists=True),
    gpx: Path | None = typer.Option(None, "--gpx"),
    camera: str = typer.Option("dashcam", "--camera", "-c"),
    fps: float = typer.Option(2.0, "--fps"),
) -> None:
    """Preflight-check your own video + GPX before spending API calls."""
    from .doctor import run_doctor

    raise typer.Exit(run_doctor(video, gpx, camera, fps))


@app.command()
def ingest(
    video: Path = typer.Option(..., "--video", "-v", exists=True, help="Input video."),
    gpx: Path | None = typer.Option(None, "--gpx", help="GPX track for this video."),
    route: Path | None = typer.Option(
        None, "--route", help="Fallback route YAML: points: [[lat, lon], ...]"
    ),
    out: Path = typer.Option(Path("run"), "--out", "-o", help="Run directory."),
    fps: float = typer.Option(2.0, "--fps", help="Keyframe sampling rate."),
    spacing_m: float = typer.Option(
        0.0,
        "--spacing-m",
        help=(
            "Sample every N METRES of travel instead of by fps. Needs a GPX. "
            "This is usually what you want: at a fixed fps a fast vehicle sees "
            "each object once, which leaves nothing to corroborate or track. "
            "Try 2.0."
        ),
    ),
    domain: str = typer.Option("road", "--domain", "-d"),
    time_offset: float = typer.Option(
        0.0, "--time-offset", help="Seconds to shift video time vs GPS time."
    ),
    heading_baseline: float = typer.Option(
        8.0,
        "--heading-baseline",
        help=(
            "Metres of travel used to derive heading. Must exceed GPS noise. "
            "8 is fine for driving; use 15-20 for WALKING, where consecutive "
            "fixes are closer together than the GPS error."
        ),
    ),
) -> None:
    """Extract keyframes and attach a position + heading to each one."""
    if gpx is None and route is None:
        _die("Provide either --gpx or --route.")

    try:
        dom = DomainConfig.load(domain)
    except FileNotFoundError as exc:
        _die(str(exc))

    out.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video) or 0.0

    # Load the track first when sampling by distance -- the GPS is what decides
    # which frames to keep.
    track_start = None
    if gpx is not None:
        track, track_start = load_gpx(gpx)
    else:
        track = load_route_yaml(route, duration)  # type: ignore[arg-type]

    if spacing_m and spacing_m > 0:
        from .ingest import extract_frames_at, plan_spacing_indices, probe_frame_times

        src_times = probe_frame_times(video)
        if not src_times:
            _die("Could not read source frame timestamps; use --fps instead.")
        idx, st = plan_spacing_indices(src_times, track, spacing_m, time_offset)
        _echo(
            f"Sampling every ~{spacing_m:g} m of travel: keeping {st['kept']} of "
            f"{st['source_frames']} source frames"
        )
        if st.get("achieved_median_m") is not None:
            _echo(f"  achieved median spacing {st['achieved_median_m']} m")
        if st.get("source_limited"):
            _echo(
                "  WARNING: the source cannot supply this density -- the vehicle "
                "moves further than the requested spacing between source frames. "
                "Positions are still correct, but objects will be seen fewer times."
            )
        frame_paths, frame_times = extract_frames_at(
            video, out / "frames", idx, src_times
        )
    else:
        _echo(f"Extracting keyframes at {fps} fps ...")
        frame_paths, frame_times = extract_frames(video, out / "frames", fps=fps)

    duration = duration or (frame_times[-1] if frame_times else 0.0)

    frames = build_frames(
        frame_paths,
        frame_times,
        track,
        time_offset=time_offset,
        track_start_iso=track_start,
        min_baseline_m=heading_baseline,
    )
    write_frames(frames, out / "frames.json")

    manifest = RunManifest(
        run_id=out.name,
        domain=dom.key,
        domain_label=dom.label,
        created=_now_iso(),
        video=str(video),
        origin=[frames[0].lat, frames[0].lon],
        n_frames=len(frames),
        duration_sec=round(duration, 2),
    )
    _save_manifest(manifest, out)

    _echo(
        f"  {len(frames)} keyframes  |  {duration:.1f}s  |  "
        f"origin {frames[0].lat:.5f}, {frames[0].lon:.5f}  |  "
        f"heading {frames[0].heading_deg:.0f}deg"
    )


@app.command()
def perceive(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    backend: str = typer.Option(
        "mock", "--backend", "-b",
        help="mock | cosmos | onnx | ensemble | locate-anything",
    ),
    truth: Path | None = typer.Option(
        None, "--truth", help="Ground-truth JSON for the mock backend."
    ),
    model: str | None = typer.Option(None, "--model", help="Override POS_MODEL."),
    model_path: Path | None = typer.Option(
        None, "--model-path", help="ONNX model file for --backend onnx/ensemble."
    ),
    limit: int = typer.Option(0, "--limit", help="Only process the first N frames."),
    tile: int = typer.Option(
        0,
        "--tile",
        help=(
            "Sliced inference for the local ONNX model: run it on overlapping "
            "N-px tiles as well as the full frame. 640 is the natural value. "
            "Recovers small distant defects that the 3x letterbox downscale "
            "destroys -- measured 2.83x more detections -- at ~8x the CPU time. "
            "Free, no API cost."
        ),
    ),
    classes_per_call: int = typer.Option(
        0,
        "--classes-per-call",
        help=(
            "Split the taxonomy into batches of N and ask about each batch "
            "separately. 0 = one open-ended pass (cheapest). Small N finds "
            "materially more, at N-times the API cost."
        ),
    ),
    workers: int = typer.Option(
        0,
        "--workers",
        "-w",
        help=(
            "Frames to run through the detector at once. 0 = auto (8 for cosmos, "
            "6 for ensemble, cores-1 for onnx, 1 for mock). 1 forces the old "
            "strictly-serial path."
        ),
    ),
) -> None:
    """Run a detector over every keyframe."""
    from .perception import default_workers, get_detector

    manifest = _load_manifest(run)
    dom = DomainConfig.load(manifest.domain)
    frames = read_frames(run / "frames.json")
    if limit > 0:
        frames = frames[:limit]

    n_workers = max(1, workers if workers > 0 else default_workers(backend))

    try:
        detector = get_detector(
            backend,
            dom,
            cache_dir=run / ".cache",
            truth_path=truth,
            model=model,
            classes_per_call=classes_per_call,
            model_path=model_path,
            tile=tile,
            # With frames already running in parallel, letting ORT also fan each
            # inference across every core just makes the workers fight for them.
            intra_op_threads=1 if n_workers > 1 else 0,
        )
    except Exception as exc:  # noqa: BLE001 - surface config errors cleanly
        _die(str(exc))

    if tile:
        _echo(
            f"  sliced inference on: full frame + {tile}px tiles. Finds smaller "
            "defects, ~8x slower on CPU."
        )

    if backend in ("cosmos", "ensemble"):
        _echo("Probing ...")
        try:
            resolved = detector.probe()  # type: ignore[attr-defined]
            _echo(f"  model: {resolved}")
        except Exception as exc:  # noqa: BLE001
            _die(f"Endpoint probe failed: {exc}")

    out_path = run / "detections.ndjson"
    total = 0

    if n_workers > 1:
        _echo(f"  {n_workers} frames in flight")

    # Flush as we go: a long paid run must survive a crash with its finished
    # frames on disk.
    with open(out_path, "w") as fh:

        def emit(dets_list: list) -> int:
            n = 0
            for dets in dets_list:
                for d in dets:
                    fh.write(json.dumps(d.model_dump()) + "\n")
                n += len(dets)
            fh.flush()
            return n

        if n_workers == 1:
            for i, frame in enumerate(frames, 1):
                total += emit([detector.detect(frame, run / frame.path)])
                if i % 10 == 0 or i == len(frames):
                    _echo(f"  {i}/{len(frames)} frames, {total} detections")
        else:
            from concurrent.futures import ThreadPoolExecutor

            # Chunked rather than one giant map, for two reasons: detections stay
            # in frame order on disk (localize and cluster both assume it), and
            # only a chunk's worth of results is ever held in memory.
            chunk = n_workers * 4
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for start in range(0, len(frames), chunk):
                    batch = frames[start : start + chunk]
                    results = list(
                        pool.map(lambda f: detector.detect(f, run / f.path), batch)
                    )
                    total += emit(results)
                    done = start + len(batch)
                    _echo(f"  {done}/{len(frames)} frames, {total} detections")

    manifest.n_detections = total
    manifest.backend = backend
    _save_manifest(manifest, run)

    hits = getattr(detector, "cache_hits", 0)
    calls = getattr(detector, "api_calls", 0)
    if calls or hits:
        _echo(f"  api calls: {calls}, cache hits: {hits}")
    _echo(f"{total} detections -> {out_path}")


@app.command()
def localize(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    camera: str = typer.Option("dashcam", "--camera", "-c"),
) -> None:
    """Project each detection onto the ground plane and give it a lat/lon."""
    cam = CameraConfig.load(camera)
    frames = read_frames(run / "frames.json")
    dets = read_detections(run / "detections.ndjson")

    # The domain decides which pixel row to project: point defects from the box
    # bottom, area/segment classes from the centre.
    dom = None
    try:
        dom = DomainConfig.load(_load_manifest(run).domain)
    except (FileNotFoundError, ValueError):
        pass

    dets = localize_detections(dets, frames, cam, dom)
    write_detections(dets, run / "detections.ndjson")
    if dom is not None:
        centred = sum(1 for d in dets if d.anchor == "centre")
        if centred:
            _echo(
                f"  {centred} area/segment detection(s) anchored at the box centre "
                "rather than its bottom edge"
            )

    ranged = [d for d in dets if d.range_m is not None]
    _echo(
        f"{len(ranged)}/{len(dets)} detections ranged on the ground plane "
        f"(camera height {cam.height_m} m, vfov {cam.vfov_deg}deg)"
    )
    if ranged:
        rs = sorted(d.range_m for d in ranged if d.range_m is not None)
        _echo(
            f"  range: min {rs[0]:.1f} m, median {rs[len(rs) // 2]:.1f} m, "
            f"max {rs[-1]:.1f} m"
        )


@app.command()
def cluster(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    camera: str = typer.Option(
        "dashcam", "--camera", "-c", help="Enables triangulation of tracked objects."
    ),
    no_triangulate: bool = typer.Option(
        False, "--no-triangulate", help="Ground-plane positions only."
    ),
) -> None:
    """Merge repeated sightings into one finding each.

    Associates detections across adjacent keyframes first, then falls back to
    spatial clustering. A tracked object seen twice from different positions is
    positioned by triangulating its bearing rays, which does not depend on camera
    pitch or the ground being flat.
    """
    manifest = _load_manifest(run)
    dom = DomainConfig.load(manifest.domain)
    frames = read_frames(run / "frames.json")
    dets = read_detections(run / "detections.ndjson")
    cam = None if no_triangulate else CameraConfig.load(camera)

    findings = cluster_detections(dets, frames, dom, cam)

    tri = [f for f in findings if f.pos_method == "triangulated"]
    if tri:
        res = sorted(f.pos_residual_m or 0.0 for f in tri)
        _echo(
            f"  {len(tri)}/{len(findings)} triangulated from {sum(f.n_rays for f in tri)} "
            f"rays, median residual {res[len(res) // 2]:.2f} m"
        )
    elif cam is not None:
        _echo("  0 findings triangulated -- no object was seen twice with enough parallax")

    # Absence findings are INFERRED from coverage, not detected. Only apply a
    # rule when the backend could plausibly have seen the asset -- otherwise
    # "we never looked" would be reported as "it is not there".
    from .absence import coverage_report, infer_absences

    detectable = _detectable_assets(manifest.backend, dom)
    absences = infer_absences(frames, findings, dom, detectable=detectable)
    if absences:
        findings = findings + absences
        for a in absences:
            _echo(f"  + {a.finding_id}  {a.label:26s} inferred from coverage")
    skipped = [r.key for r in dom.absences if r.asset not in detectable]
    if skipped:
        _echo(f"  absence rules skipped (asset not detectable by "
              f"{manifest.backend}): {', '.join(skipped)}")

    write_findings(findings, run / "findings.json")
    (run / "coverage.json").write_text(
        json.dumps(coverage_report(frames, findings, dom), indent=2)
    )

    manifest.n_findings = len(findings)
    _save_manifest(manifest, run)

    _echo(f"{len(dets)} detections -> {len(findings)} findings")
    for f in findings[:8]:
        loc = f"{f.lat:.5f},{f.lon:.5f}" if f.lat is not None else "unlocated"
        _echo(
            f"  {f.finding_id}  {f.label:26s} sev {f.severity}  "
            f"conf {f.confidence:.2f}  {len(f.evidence)}x  {loc}"
        )
    if len(findings) > 8:
        _echo(f"  ... and {len(findings) - 8} more")


@app.command()
def score(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    segment_m: float = typer.Option(20.0, "--segment-m"),
) -> None:
    """Score the route in segments and write the summary."""
    manifest = _load_manifest(run)
    dom = DomainConfig.load(manifest.domain)
    frames = read_frames(run / "frames.json")
    findings = read_findings(run / "findings.json")

    segments = build_segments(frames, findings, dom, segment_m=segment_m)
    write_segments(segments, run / "segments.json")
    summary = summarize(findings, segments, dom)

    manifest.summary = summary
    _save_manifest(manifest, run)

    _echo(f"{dom.index_name}: {summary.quality_index}/100  (grade {summary.grade})")
    _echo(f"  {len(segments)} segments over {summary.route_length_m} m")
    for cls, n in list(summary.counts.items())[:10]:
        _echo(f"  {n:3d}x  {dom.spec(cls).label}")


@app.command()
def twin(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    margin_m: float = typer.Option(120.0, "--margin-m", help="Bbox padding."),
    offline: bool = typer.Option(False, "--offline", help="Skip the Overpass call."),
) -> None:
    """Fetch OpenStreetMap context geometry around the route."""
    from .twin import build_twin

    manifest = _load_manifest(run)
    frames = read_frames(run / "frames.json")
    try:
        tw = build_twin(frames, run / "twin.json", margin_m=margin_m, offline=offline)
    except Exception as exc:  # noqa: BLE001 - Overpass is rate limited, not broken
        # One clean line, not a traceback: this runs as a subprocess whose output is
        # streamed into the studio's browser log. Nothing was written, so the fix is
        # to run the same command again in a minute.
        _die(
            f"Overpass unavailable ({type(exc).__name__}: {exc}). "
            f"No twin.json written -- re-run `pos twin --run {run}` to retry. "
            f"The viewer works without it, minus the 3D buildings layer."
        )

    manifest.has_twin = bool(tw.buildings or tw.roads)
    _save_manifest(manifest, run)
    _echo(
        f"{len(tw.buildings)} buildings, {len(tw.roads)} roads -> {run / 'twin.json'}"
    )


@app.command()
def pointcloud(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    preds: Path = typer.Option(
        ..., "--preds", exists=True, help="Directory of lingbot-map *.npz predictions."
    ),
    budget: int = typer.Option(400_000, "--budget", help="Max points to keep."),
    conf: float = typer.Option(1.5, "--conf", help="Confidence threshold."),
    raw: bool = typer.Option(
        False, "--raw", help="Skip georeferencing (keep reconstruction coordinates)."
    ),
) -> None:
    """Convert lingbot-map predictions into run/cloud.ply.

    See scripts/lingbot_gpu_pass.sh for producing the predictions on a rented GPU.
    """
    from .pointcloud import build_pointcloud

    manifest = _load_manifest(run)
    frames = read_frames(run / "frames.json")

    n = build_pointcloud(
        preds,
        frames,
        run / "cloud.ply",
        origin=(manifest.origin[0], manifest.origin[1]),
        budget=budget,
        conf_threshold=conf,
        georeference=not raw,
    )

    manifest.has_pointcloud = True
    _save_manifest(manifest, run)
    _echo(f"{n:,} points -> {run / 'cloud.ply'}")


@app.command()
def depthcloud(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    camera: str = typer.Option("dashcam", "--camera", "-c"),
    stride: int = typer.Option(6, "--stride", help="Pixel sampling step."),
    budget: int = typer.Option(500_000, "--budget", help="Max points to keep."),
    max_range: float = typer.Option(30.0, "--max-range"),
    model_path: Path | None = typer.Option(None, "--model-path"),
) -> None:
    """Build run/cloud.ply from the video on CPU, using monocular depth.

    No GPU needed. Depth Anything V2 gives relative depth; the ground plane from
    your camera calibration fixes it to metres. For a better reconstruction use
    `pos pointcloud` with a lingbot-map GPU pass instead.
    """
    from .depthcloud import DepthCloudError, build_depth_cloud

    manifest = _load_manifest(run)
    cam = CameraConfig.load(camera)
    frames = read_frames(run / "frames.json")

    _echo(f"Monocular depth over {len(frames)} keyframes (CPU, ~1.5 s each) ...")
    _echo(f"  camera {camera}: height {cam.height_m} m, vfov {cam.vfov_deg}deg")
    try:
        n = build_depth_cloud(
            frames,
            run,
            cam,
            (manifest.origin[0], manifest.origin[1]),
            run / "cloud.ply",
            stride=stride,
            budget=budget,
            max_range_m=max_range,
            model_path=model_path,
        )
    except DepthCloudError as exc:
        _die(str(exc))

    manifest.has_pointcloud = True
    _save_manifest(manifest, run)
    _echo(f"{n:,} points -> {run / 'cloud.ply'}")
    _echo("Toggle the 'Point cloud' layer in the viewer to see it.")


@app.command()
def report(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    out: Path | None = typer.Option(None, "--out", "-o", help="Defaults to <run>.pdf"),
    image_width: int = typer.Option(
        1100,
        "--image-width",
        help="Embedded photo width in px. Drop to ~700 for a mailable file.",
    ),
) -> None:
    """Write a paginated PDF inspection report.

    Cover with the score and asset coverage, the segment table worst-first, then a
    page per finding with its photograph, coordinates and the model's own reason.

    Every page footer names the perception backend, so a reader can tell real
    detections from `mock` synthetic fixtures. A report that looked authoritative
    while hiding that would launder a guess into a record.
    """
    from .report import build_report

    target = Path(out) if out else Path(f"{Path(run).name}.pdf")
    info = build_report(run, target, image_width=image_width)
    _echo(f"{info['bytes'] / 1e6:.2f} MB -> {info['path']}")
    if info["bytes"] > 25e6:
        _echo(
            "  over 25 MB -- most mail providers will reject it. "
            "Re-run with --image-width 700 to roughly halve it."
        )
    _echo(
        f"  {info['pages']} pages: cover + "
        f"{'segment table + ' if info['segments'] else ''}"
        f"{info['finding_pages']} finding page(s)"
    )


@app.command()
def verify(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    model: str | None = typer.Option(None, "--model"),
    segment_m: float = typer.Option(20.0, "--segment-m"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report verdicts without rewriting findings."
    ),
) -> None:
    """Re-examine the shakier findings and drop the ones that do not hold up.

    Single-pass perception measures 64% precision. Findings seen twice or more
    already run at ~83%, so this only re-checks SINGLETONS with sub-0.85
    confidence -- roughly 20-30 calls on a 57-finding run, not 57.

    Each candidate's clearest sighting is cropped and re-shown to the model with a
    sceptical prompt that names the lookalikes to reject. Rejected findings are
    removed; unconfirmed ones stay with reduced confidence. Absence findings are
    never checked, since they have no pixels to inspect.

    Rescores afterwards, because dropping findings without rescoring would leave
    an index that disagrees with its own evidence.
    """
    from .perception.cosmos import CosmosDetector, CosmosError
    from .perception.verify import verify_findings

    manifest = _load_manifest(run)
    dom = DomainConfig.load(manifest.domain)
    findings = read_findings(run / "findings.json")
    frames = read_frames(run / "frames.json")

    try:
        detector = CosmosDetector(dom, model=model, cache_dir=run / ".cache")
    except CosmosError as exc:
        _die(f"{exc}\nVerification needs a hosted VLM -- set NVIDIA_API_KEY in .env")

    _echo(f"Verifying with {detector.model} ...")
    kept, verdicts, stats = verify_findings(findings, run, detector, dom)

    _echo(
        f"  checked {stats['checked']} of {stats['input']}: "
        f"{stats['confirmed']} confirmed, {stats['rejected']} rejected, "
        f"{stats['unsure']} unsure"
    )
    if stats["skipped_no_image"]:
        _echo(f"  {stats['skipped_no_image']} skipped: evidence frame missing")
    _echo(f"  {detector.api_calls} API calls, {detector.cache_hits} cache hits")

    for v in verdicts:
        if v.verdict == "reject":
            _echo(f"  - dropped {v.finding_id} ({v.cls}): {v.reason or 'rejected'}")

    # `dry_run` must be recorded: without it the report would state that
    # rejected findings were "removed" when findings.json was never touched.
    (run / "verification.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "model": detector.model,
                "dry_run": dry_run,
                "stats": stats,
                "verdicts": [v.__dict__ for v in verdicts],
            },
            indent=2,
        )
    )

    if dry_run:
        _echo("\n  --dry-run: findings.json unchanged.")
        return

    (run / "findings.json").write_text(
        json.dumps([json.loads(f.model_dump_json()) for f in kept], indent=2)
    )

    # Rescore: the index is derived from findings, so it must move with them.
    segments = build_segments(frames, kept, dom, segment_m=segment_m)
    write_segments(segments, run / "segments.json")
    summary = summarize(kept, segments, dom)
    manifest.n_findings = len(kept)
    manifest.summary = summary
    _save_manifest(manifest, run)

    _echo(
        f"\n  {stats['input']} -> {len(kept)} findings   "
        f"{dom.index_name} {summary.quality_index}/100 (grade {summary.grade})"
    )


@app.command()
def basemap(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    provider: str = typer.Option("esri", "--provider", "-p", help="esri | osm | mapbox"),
    zoom: int = typer.Option(18, "--zoom", "-z", help="18 ≈ 0.5 m/px."),
    margin: float = typer.Option(60.0, "--margin", help="Padding around the route, m."),
    tiles: str | None = typer.Option(None, "--tiles", help="Custom {z}/{x}/{y} URL."),
    attribution: str | None = typer.Option(None, "--attribution"),
    token: str | None = typer.Option(None, "--token", help="For mapbox."),
) -> None:
    """Fetch satellite imagery under the route and cache it in the run.

    Turns the dark grey ground plane into the actual place, which is what lets a
    viewer judge whether a finding really sits on the road.

    LICENCE: imagery is not public domain. `esri` (the default) is free for
    NON-COMMERCIAL use and needs attribution, which the viewer renders from
    basemap.json -- do not strip it. Google's tiles are not offered because they
    may not be used outside Google's own APIs; use `pos kml` for those.
    """
    from .basemap import PROVIDERS, BasemapError, build_basemap

    manifest = _load_manifest(run)
    frames = read_frames(run / "frames.json")

    preset = PROVIDERS.get(provider, {})
    if preset.get("licence"):
        _echo(f"  {provider}: {preset['licence']}")

    _echo(f"Fetching {provider} tiles at z{zoom} ...")
    try:
        meta = build_basemap(
            frames,
            (manifest.origin[0], manifest.origin[1]),
            run,
            provider=provider,
            zoom=zoom,
            margin_m=margin,
            url_template=tiles,
            attribution=attribution,
            token=token,
        )
    except BasemapError as exc:
        _die(str(exc))

    t = meta["tiles"]
    _echo(
        f"  {t['fetched']}/{t['nx'] * t['ny']} tiles"
        + (f", {t['failed']} failed" if t["failed"] else "")
    )
    _echo(
        f"  {meta['size']['w']}x{meta['size']['h']} px, {meta['m_per_px']:.2f} m/px, "
        f"covering {meta['local']['width_m']:.0f} x {meta['local']['height_m']:.0f} m"
    )
    _echo(f"  {meta['attribution']}")
    if meta["kind"] != "satellite":
        _echo(f"  NOTE: {provider} is a {meta['kind']} map, not satellite imagery.")
    _echo("Toggle the 'Satellite' layer in the viewer.")


@app.command()
def import_csv(
    csv_file: Path = typer.Argument(..., exists=True, help="Path to the defects CSV file."),
    run: Path = typer.Option(Path("run"), "--run", "-r", help="Run directory to create or update."),
) -> None:
    """Import an external CSV into the dashboard.
    
    Reads a CSV with columns: object_id, frame, time, defect_name, lat, lon, frame_url
    and generates findings.json (plus manifest.json and frames.json if they don't exist).
    Downloads Google Drive images locally so the dashboard can view them.
    """
    from .import_csv import run_import_csv
    run_import_csv(csv_file, run)


@app.command()
def export_csv(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    out: Path | None = typer.Option(None, "--out", "-o", help="Defaults to <run>/defects.csv"),
    base_url: str = typer.Option("", "--base-url", help="Base URL for frame_url. If blank, uses frames/<frame_id>.jpg"),
) -> None:
    """Export findings as a CSV with chainage and coordinates.

    severity and confidence are the last two columns, and they are what makes the file
    round-trip: the quality index is weight x severity x confidence, so a CSV without
    them re-imports as a uniform severity 3 at confidence 1.0 and scores differently
    from the run it came from. `pos import-csv` reads them when present and falls back
    to those defaults when absent, so a hand-written eight-column file still works.
    """
    import csv
    from .geo import haversine_m
    
    target = Path(out) if out else run / "defects.csv"
    manifest = _load_manifest(run)
    try:
        dom = DomainConfig.load(manifest.domain)
    except FileNotFoundError:
        dom = DomainConfig(key="road", label="Road")
        
    findings = read_findings(run / "findings.json")
    frames = read_frames(run / "frames.json")
    
    # Calculate cumulative chainage for each frame
    frame_chainage = {}
    current_m = 0.0
    for i in range(len(frames)):
        if i > 0:
            current_m += haversine_m(frames[i-1].lat, frames[i-1].lon, frames[i].lat, frames[i].lon)
        frame_chainage[frames[i].frame_id] = current_m
        
    def _best_sighting(evidence):
        if not evidence:
            return None
        return max(evidence, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        
    with open(target, "w", newline="") as f:
        writer = csv.writer(f)
        # severity/confidence appended, not inserted: a reader keyed on the original
        # eight column NAMES keeps working, and so does one keyed on position.
        writer.writerow(["object_id", "frame", "time", "defect_name", "lat", "lon",
                         "chainage_km", "frame_url", "severity", "confidence"])
        
        for idx, finding in enumerate(sorted(findings, key=lambda x: x.t_sec)):
            det = _best_sighting(finding.evidence)
            frame_id = det.frame_id if det else ""
            chainage_km = frame_chainage.get(frame_id, 0.0) / 1000.0 if frame_id else 0.0
            
            # Sub-second precision, not int(): scoring assigns a finding to the segment
            # whose TIME window contains it, so truncating to a whole second slides it
            # up to ~5 m along the route at survey speed and can move it into the
            # neighbouring 20 m segment. That changes the quality index on re-import
            # while every finding still looks correctly placed on the map.
            # Stays human-readable, and parse_time already accepts "0:01.839".
            m, s = divmod(finding.t_sec, 60.0)
            time_str = f"{int(m)}:{s:06.3f}"
            
            spec = dom.class_map.get(finding.cls)
            defect_name = finding.label or (spec.label if spec else finding.cls)
            
            lat = f"{finding.lat:.8f}" if finding.lat is not None else ""
            lon = f"{finding.lon:.8f}" if finding.lon is not None else ""
            
            frame_url = f"{base_url}{frame_id}.jpg" if frame_id and base_url else (f"frames/{frame_id}.jpg" if frame_id else "")
            
            writer.writerow([
                idx + 1,             # object_id (1-indexed counter like in the example)
                frame_id,            # frame
                time_str,            # time
                defect_name,         # defect_name
                lat,                 # lat
                lon,                 # lon
                f"{chainage_km:.3f}",# chainage_km
                frame_url,           # frame_url
                finding.severity,    # severity   1-5, drives the quality index
                f"{finding.confidence:.3f}",  # confidence 0-1, scales the penalty
            ])
            
    _echo(f"Exported {len(findings)} findings to {target}")


@app.command()
def kml(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    out: Path | None = typer.Option(None, "--out", "-o", help="Defaults to <run>.kmz"),
    no_tour: bool = typer.Option(False, "--no-tour", help="Skip the flythrough."),
    tour_stops: int = typer.Option(40, "--tour-stops"),
) -> None:
    """Export the run as a KMZ for Google Earth.

    One self-contained file: route, findings with their evidence photos, the
    quality heatmap, missing-asset stretches, and a flythrough. No API key and no
    network -- evidence images and marker icons are packed inside the archive.

    Open it in Google Earth Pro (free) or Earth Web. Google Earth supplies real
    satellite imagery and worldwide 3D terrain, which is the point: it shows
    whether a finding really sits on the road. Note that Google's photorealistic
    3D BUILDINGS cover about 2,500 cities only, so rural routes get imagery and
    terrain but no 3D city model.

    Google Earth Pro's Tools > Movie Maker records the flythrough straight to
    MP4, which beats screen-capturing a browser for a demo video.
    """
    from .kmlexport import build_kmz

    target = Path(out) if out else Path(f"{Path(run).name}.kmz")
    info = build_kmz(run, target, tour=not no_tour, max_tour_stops=tour_stops)

    _echo(f"{info['bytes'] / 1e6:.1f} MB -> {info['path']}")
    _echo(
        f"  {info['placemarks']} findings, {info['segments']} heatmap segments, "
        f"{info['gap_lines']} missing-asset stretch(es), {info['images']} photos"
    )
    if info["tour_stops"]:
        _echo(f"  flythrough with {info['tour_stops']} stops")
    if info["has_track"]:
        _echo("  vehicle track present -- Google Earth will show its time slider")
    if info["skipped"]:
        _echo(f"  {info['skipped']} finding(s) skipped: no position and no frame")


@app.command("run")
def run_all(
    video: Path = typer.Option(..., "--video", "-v", exists=True),
    gpx: Path | None = typer.Option(None, "--gpx"),
    route: Path | None = typer.Option(None, "--route"),
    out: Path = typer.Option(Path("run"), "--out", "-o"),
    domain: str = typer.Option("road", "--domain", "-d"),
    backend: str = typer.Option("mock", "--backend", "-b"),
    camera: str = typer.Option("dashcam", "--camera", "-c"),
    fps: float = typer.Option(2.0, "--fps"),
    spacing_m: float = typer.Option(
        0.0,
        "--spacing-m",
        help="Sample every N metres of travel instead of by fps. Needs a GPX. Try 2.0.",
    ),
    no_triangulate: bool = typer.Option(False, "--no-triangulate"),
    truth: Path | None = typer.Option(None, "--truth"),
    model_path: Path | None = typer.Option(None, "--model-path"),
    segment_m: float = typer.Option(20.0, "--segment-m"),
    tile: int = typer.Option(
        0, "--tile", help="Sliced inference for the ONNX model. Try 640."
    ),
    classes_per_call: int = typer.Option(0, "--classes-per-call"),
    workers: int = typer.Option(
        0, "--workers", "-w",
        help="Frames detected concurrently. 0 = auto per backend, 1 = serial.",
    ),
    time_offset: float = typer.Option(
        0.0,
        "--time-offset",
        help="Seconds to shift video time vs GPS time. Get it from `pos doctor`.",
    ),
    heading_baseline: float = typer.Option(
        8.0, "--heading-baseline",
        help="Metres of travel used for heading. Use 15-20 when WALKING.",
    ),
    skip_twin: bool = typer.Option(False, "--skip-twin"),
    with_segment: bool = typer.Option(
        False, "--segment/--no-segment",
        help="Also segment the carriageway in every keyframe (writes roadseg.json).",
    ),
    # NOT --video: `run` already takes --video/-v for the source clip, and typer
    # will happily register a second option with the same name, after which one
    # of the two silently stops working.
    with_video: bool = typer.Option(
        False, "--render-video/--no-render-video",
        help="Also render the annotated review MP4. Implies --segment for the overlay.",
    ),
    video_stride: int = typer.Option(
        1, "--video-stride", help="Render every Nth keyframe into the video."
    ),
) -> None:
    """Run the whole pipeline: ingest -> perceive -> localize -> cluster -> score."""
    _echo("=== ingest ===")
    ingest(
        video=video,
        gpx=gpx,
        route=route,
        out=out,
        fps=fps,
        spacing_m=spacing_m,
        domain=domain,
        time_offset=time_offset,
        heading_baseline=heading_baseline,
    )
    _echo("\n=== perceive ===")
    perceive(
        run=out, backend=backend, truth=truth, model=None, limit=0,
        classes_per_call=classes_per_call, model_path=model_path, tile=tile,
        workers=workers,
    )
    _echo("\n=== localize ===")
    localize(run=out, camera=camera)
    _echo("\n=== cluster ===")
    cluster(run=out, camera=camera, no_triangulate=no_triangulate)
    _echo("\n=== score ===")
    score(run=out, segment_m=segment_m)

    if not skip_twin:
        _echo("\n=== twin ===")
        try:
            twin(run=out, margin_m=120.0, offline=False)
        except Exception as exc:  # noqa: BLE001 - Overpass is flaky, never fatal
            typer.secho(
                f"  twin skipped ({exc}). The viewer works without it.",
                fg=typer.colors.YELLOW,
            )

    # Segmentation runs AFTER scoring, not before: nothing upstream consumes the
    # road mask yet, and putting a second ONNX model in front of perception
    # would delay the findings -- the part of the run someone is waiting for.
    if with_segment or with_video:
        _echo("\n=== segment ===")
        try:
            segment(run=out, camera=camera, stride=1, workers=0, model_path=None)
        except Exception as exc:  # noqa: BLE001 - a missing road model is not fatal
            typer.secho(f"  segmentation skipped ({exc}).", fg=typer.colors.YELLOW)

    if with_video:
        _echo("\n=== video ===")
        try:
            video(
                run=out, out=None, stride=video_stride, limit=0,
                road=True, model_path=None,
            )
        except Exception as exc:  # noqa: BLE001 - the run is already complete
            typer.secho(f"  video skipped ({exc}).", fg=typer.colors.YELLOW)

    _echo(f"\nDone. Serve it:  uv run pos serve --run {out}")


@app.command()
def segment(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    camera: str = typer.Option("dashcam", "--camera", "-c"),
    stride: int = typer.Option(1, "--stride", help="Segment every Nth keyframe."),
    workers: int = typer.Option(
        0, "--workers", "-w",
        help="Frames segmented concurrently. 0 = auto (cores-1, capped at 8).",
    ),
    model_path: Path | None = typer.Option(None, "--model-path", help="Road UNet .onnx."),
) -> None:
    """Segment the drivable carriageway in every keyframe.

    Writes roadseg.json: per-frame road coverage, the carriageway polygon, and a
    width cross-section. Two things downstream want it -- `pos video` draws the
    mask, and on-carriageway gating uses the polygon to tell a pothole in the
    road from one detected on a spoil heap beside it.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    from .roadzone import carriageway_width, width_summary
    from .segment import RoadSegmenter

    _load_manifest(run)  # fail early and consistently if the run is malformed
    cam = CameraConfig.load(camera)
    frames = read_frames(run / "frames.json")[:: max(stride, 1)]
    if not frames:
        _die("no keyframes to segment")

    n_workers = max(1, workers if workers > 0 else min(8, (os.cpu_count() or 2) - 1))
    # One session shared across the pool, one intra-op thread each: ORT sessions
    # are safe to call concurrently, and a session per worker would multiply the
    # model's memory by the worker count for nothing. Same trade-off as perceive.
    seg = RoadSegmenter(model_path=model_path, intra_op_threads=1 if n_workers > 1 else 0)
    _echo(f"  {n_workers} frames in flight, camera {camera}")

    def one(frame):
        mask = seg.mask(run / frame.path, (frame.width, frame.height))
        poly = seg.polygon(mask)
        widths = carriageway_width(mask, cam, frame_id=frame.frame_id)
        row = {
            "frame_id": frame.frame_id,
            "t_sec": frame.t_sec,
            "coverage": round(seg.coverage(mask), 4),
            "polygon": [] if poly is None else poly.reshape(-1, 2).astype(int).tolist(),
            "widths": [
                {
                    "range_m": s.range_m,
                    "row": s.row,
                    "span_m": s.span_m,
                    "free_m": s.free_m,
                    "ok": s.ok,
                    "note": s.note,
                }
                for s in widths.samples
            ],
        }
        return row, widths

    rows: list[dict] = []
    all_widths = []
    chunk = n_workers * 4
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for start in range(0, len(frames), chunk):
            batch = frames[start : start + chunk]
            for row, w in pool.map(one, batch):
                rows.append(row)
                all_widths.append(w)
            _echo(f"  {start + len(batch)}/{len(frames)} frames")

    summary = width_summary(all_widths)
    covs = sorted(r["coverage"] for r in rows if r["coverage"] > 0)
    found = len(covs)

    (run / "roadseg.json").write_text(
        json.dumps(
            {
                "camera": camera,
                "stride": stride,
                "n_frames": len(rows),
                "n_with_road": found,
                "coverage_median": round(covs[len(covs) // 2], 4) if covs else 0.0,
                "width_summary": {
                    "n_usable": summary.n_usable,
                    "n_clipped": summary.n_clipped,
                    "n_no_road": summary.n_no_road,
                    "median_span_m": summary.median_span_m,
                    "median_free_m": summary.median_free_m,
                    "min_span_m": summary.min_span_m,
                    "min_at_frame": summary.min_at_frame,
                },
                "frames": rows,
            },
            indent=1,
        )
        + "\n"
    )

    _echo(
        f"road found in {found}/{len(rows)} frames"
        + (f", median coverage {covs[len(covs) // 2] * 100:.1f}% of frame" if covs else "")
    )
    if summary.n_usable:
        _echo(
            f"carriageway width: median {summary.median_span_m} m "
            f"(free {summary.median_free_m} m), narrowest {summary.min_span_m} m "
            f"at frame {summary.min_at_frame}"
        )
    else:
        # Not a soft failure worth hiding. Width is measured by converting a
        # pixel row to a forward range through the camera model, so if no
        # sampled row lands on road at all, the calibration disagrees with the
        # footage -- and every range in the run rests on that same calibration.
        _echo(
            f"carriageway width: NO usable cross-sections "
            f"({summary.n_no_road} rows found no road, {summary.n_implausible} "
            f"produced an implausible width, {summary.n_clipped} were clipped by "
            f"the frame edge). Those rows are placed by the '{camera}' "
            f"calibration; if they miss the road or imply a 20 cm carriageway, "
            f"that calibration does not match this footage -- and every range in "
            f"the run rests on it."
        )
    _echo(f"-> {run / 'roadseg.json'}")


@app.command()
def video(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    out: Path | None = typer.Option(None, "--out", "-o", help="Defaults to <run>/review.mp4."),
    stride: int = typer.Option(1, "--stride", help="Render every Nth keyframe."),
    limit: int = typer.Option(0, "--limit", help="Stop after N rendered frames."),
    road: bool = typer.Option(True, "--road/--no-road", help="Overlay the road mask."),
    model_path: Path | None = typer.Option(None, "--model-path", help="Road UNet .onnx."),
) -> None:
    """Render the drive as an annotated MP4: road mask, tracked defects, HUD.

    The one output you can watch. Everything else the pipeline makes is a map, a
    document or a still crop, and none of those show a defect being approached,
    tracked and passed -- which is what tells a reviewer the sighting was real.
    """
    from .reviewvideo import render_review

    manifest = _load_manifest(run)
    dom = DomainConfig.load(manifest.domain)
    target = out or (run / "review.mp4")

    seg = None
    if road:
        from .segment import RoadSegmenter

        try:
            seg = RoadSegmenter(model_path=model_path, intra_op_threads=0)
        except Exception as exc:  # noqa: BLE001 - the video is worth having regardless
            _echo(f"  no road overlay: {exc}")

    _echo(f"rendering {target} ...")
    result = render_review(run, target, domain=dom, stride=stride, limit=limit, segmenter=seg)

    _echo(
        f"{result.frames} frames at {result.fps:.1f} fps "
        f"({result.duration_s:.0f}s), {result.resolution[0]}x{result.resolution[1]}, "
        f"{result.codec}"
    )
    if result.codec != "h264":
        _echo("  WARNING: ffmpeg not found, so this is mp4v -- browsers will not play it.")
    _echo(f"  {result.detections_drawn} detections drawn, road on {result.frames_with_road} frames")
    _echo(f"-> {result.path}")


@app.command()
def serve(
    run: Path = typer.Option(Path("run"), "--run", "-r", exists=True),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    studio: bool = typer.Option(
        False,
        "--studio",
        help="Also serve the upload page at /studio and allow browsing any run.",
    ),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    uploads_dir: Path = typer.Option(Path("uploads"), "--uploads-dir"),
) -> None:
    """Serve a run directory plus the SSE alert stream.

    With --studio it also serves an upload page: drop in a video and its GPX,
    watch the pipeline run, then open the result. Any run under --runs-dir
    becomes viewable via `?run=<name>`, so a new upload needs no restart.
    """
    from .server import serve as _serve

    if studio:
        runs_dir.mkdir(parents=True, exist_ok=True)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        _serve(run, host=host, port=port, runs_dir=runs_dir, uploads_dir=uploads_dir)
    else:
        _serve(run, host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(app())
