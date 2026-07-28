"""Build a georeferenced point cloud from the video alone, on CPU.

WHY THIS EXISTS ALONGSIDE pointcloud.py
pointcloud.py converts lingbot-map's output, which is the better reconstruction
-- but it needs a CUDA GPU, so it means renting a box and doing an offline pass.
This route needs nothing but the video, the GPS track and the camera calibration
you already have, and runs at roughly 1.5 s per keyframe on CPU.

THE TRICK: MAKING RELATIVE DEPTH METRIC
Depth Anything V2 predicts INVERSE depth up to an unknown affine transform --
"this pixel is nearer than that one", not "this pixel is 7 m away". Normally
fixing that scale needs a second sensor.

We already have one: the ground plane. For any pixel below the horizon,
pos/geo.py gives the true forward distance from calibration alone. So sample
ground pixels, pair each one's predicted inverse depth `d` with its known true
distance Z, and fit

    1/Z  =  a * d + b

by least squares -- two unknowns, hundreds of samples. Applying that fit to
every pixel converts the whole map to metres.

Each pixel then back-projects to camera coordinates, rotates by the frame's
heading and offsets by its GPS fix, so all frames land in one local ENU frame.

HONEST LIMITS
  - Monocular depth on a textureless wet road is weak. Expect a noisy surface,
    not survey geometry.
  - The affine fit assumes the ground is flat across the frame. On Kohima's
    hills that holds only locally.
  - Sky and far pixels are dropped: their inverse depth approaches the fit's
    zero crossing, where 1/Z explodes.
  - Overlapping frames re-observe the same ground, so the cloud thickens rather
    than converging. Voxelising hides most of it. This is not SLAM.

Licence: Depth Anything V2 is Apache-2.0, unlike the AGPL YOLO weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CameraConfig
from .geo import focal_px, horizon_v, latlon_to_local_m
from .pointcloud import voxel_downsample, write_ply
from .schema import Frame

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "physicalos" / "depth"
MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX/"
    "resolve/main/onnx/model_quantized.onnx"
)
INPUT_SIZE = 518  # Depth Anything V2 wants a multiple of 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthCloudError(RuntimeError):
    pass


def ensure_model(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Download the depth model on first use. Weights live in a sidecar file."""
    model_dir.mkdir(parents=True, exist_ok=True)
    # Keep the UPSTREAM filename. The ONNX graph embeds a literal reference to
    # its weight sidecar ("location: model_quantized.onnx_data"), so renaming the
    # model breaks loading with a confusing "External data path does not exist".
    onnx = model_dir / "model_quantized.onnx"
    data = model_dir / "model_quantized.onnx_data"

    if onnx.exists() and data.exists():
        return onnx

    try:
        import httpx
    except ImportError as exc:
        raise DepthCloudError("httpx is required to download the depth model") from exc

    for url, dest in ((MODEL_URL, onnx), (MODEL_URL + "_data", data)):
        print(f"  downloading {dest.name} ...")
        with httpx.stream("GET", url, follow_redirects=True, timeout=600.0) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
    return onnx


def _preprocess(img) -> np.ndarray:
    r = img.resize((INPUT_SIZE, INPUT_SIZE), 3)  # 3 = BICUBIC
    x = np.asarray(r, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None]


def fit_metric_scale(
    inv_depth: np.ndarray, cam: CameraConfig, w: int, h: int
) -> tuple[float, float, int]:
    """Fit 1/Z = a*d + b using ground pixels whose true Z we can compute."""
    f = focal_px(h, cam.vfov_deg)
    hz = horizon_v(h, cam)

    rows = np.linspace(hz + 12, h - 2, 60)
    cols = np.linspace(w * 0.2, w * 0.8, 18)

    ds: list[float] = []
    inv_zs: list[float] = []
    for v in rows:
        dy = v - hz
        if dy <= 1e-6:
            continue
        z = cam.height_m * f / dy          # true forward distance for this row
        if z <= 0 or z > cam.max_range_m:
            continue
        my = int(np.clip(v / h * inv_depth.shape[0], 0, inv_depth.shape[0] - 1))
        for u in cols:
            mx = int(np.clip(u / w * inv_depth.shape[1], 0, inv_depth.shape[1] - 1))
            ds.append(float(inv_depth[my, mx]))
            inv_zs.append(1.0 / z)

    if len(ds) < 40:
        raise DepthCloudError(
            "Too few usable ground pixels to fit depth scale. Check the camera "
            "calibration -- if the horizon is wrong, no row maps to a real range."
        )

    d_arr = np.asarray(ds, dtype=np.float64)
    y_arr = np.asarray(inv_zs, dtype=np.float64)
    A = np.stack([d_arr, np.ones_like(d_arr)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    return float(a), float(b), len(ds)


def frame_points(
    img,
    inv_depth: np.ndarray,
    frame: Frame,
    cam: CameraConfig,
    origin: tuple[float, float],
    stride: int = 6,
    max_range_m: float = 30.0,
    min_range_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one frame's depth map into local ENU metres."""
    w, h = frame.width, frame.height
    f = focal_px(h, cam.vfov_deg)
    cx, hz = w / 2.0, horizon_v(h, cam)

    a, b, _ = fit_metric_scale(inv_depth, cam, w, h)

    # Sample a grid rather than all 2 Mpx: the cloud is voxelised later anyway,
    # and this keeps a 45-frame run to seconds instead of minutes.
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    gx, gy = np.meshgrid(xs, ys)

    my = np.clip((gy / h * inv_depth.shape[0]).astype(int), 0, inv_depth.shape[0] - 1)
    mx = np.clip((gx / w * inv_depth.shape[1]).astype(int), 0, inv_depth.shape[1] - 1)
    d = inv_depth[my, mx]

    inv_z = a * d + b
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(inv_z > 1e-6, 1.0 / inv_z, np.nan)

    ok = np.isfinite(z) & (z >= min_range_m) & (z <= max_range_m)
    if not ok.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    zz = z[ok]
    uu = gx[ok].astype(np.float64)
    vv = gy[ok].astype(np.float64)

    # Camera frame: X right, Y up from the ground, Z forward. A point at forward
    # Z and height Y images at v = hz + f*(height_m - Y)/Z.
    x_cam = (uu - cx) * zz / f
    y_up = cam.height_m - (vv - hz) * zz / f

    # Rotate into ENU by heading, then offset by the frame's GPS fix.
    psi = np.radians(frame.heading_deg)
    east = x_cam * np.cos(psi) + zz * np.sin(psi)
    north = -x_cam * np.sin(psi) + zz * np.cos(psi)

    fe, fn = latlon_to_local_m(frame.lat, frame.lon, origin[0], origin[1])

    # store.ts maps +X east and +Z south, so z = -north.
    pts = np.stack(
        [
            (fe + east).astype(np.float32),
            y_up.astype(np.float32),
            (-(fn + north)).astype(np.float32),
        ],
        axis=1,
    )

    rgb = np.asarray(img, dtype=np.float32)
    cols = rgb[gy[ok], gx[ok]]
    return pts, cols


def build_depth_cloud(
    frames: list[Frame],
    run_dir: Path,
    cam: CameraConfig,
    origin: tuple[float, float],
    out_path: Path,
    stride: int = 6,
    budget: int = 500_000,
    max_range_m: float = 30.0,
    model_path: Path | None = None,
    progress: bool = True,
) -> int:
    """Run depth over every keyframe and write one fused PLY."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise DepthCloudError("onnxruntime is required: uv add onnxruntime") from exc
    from PIL import Image

    path = Path(model_path) if model_path else ensure_model()
    from .ortproviders import providers

    sess = ort.InferenceSession(str(path), providers=providers())
    in_name = sess.get_inputs()[0].name

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    failed = 0

    for i, fr in enumerate(frames, 1):
        fp = run_dir / fr.path
        if not fp.exists():
            continue
        with Image.open(fp) as im:
            img = im.convert("RGB")
            out = sess.run(None, {in_name: _preprocess(img)})[0]
            inv = out[0] if out.ndim == 3 else out
            try:
                pts, cols = frame_points(
                    img, inv, fr, cam, origin,
                    stride=stride, max_range_m=max_range_m,
                )
            except DepthCloudError:
                failed += 1
                continue
        if len(pts):
            all_pts.append(pts)
            all_cols.append(cols)
        if progress and (i % 10 == 0 or i == len(frames)):
            print(f"  {i}/{len(frames)} frames, {sum(len(p) for p in all_pts):,} points")

    if not all_pts:
        raise DepthCloudError("No points produced. Check calibration and frames.")
    if failed:
        print(f"  {failed} frame(s) skipped: depth scale could not be fitted")

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    pts, cols = voxel_downsample(pts, cols, budget=budget)
    write_ply(
        pts, cols, out_path,
        comment=(
            "PhysicalOS: CPU monocular depth (Depth Anything V2 small), "
            "scaled to metres via the calibrated ground plane"
        ),
    )
    return int(pts.shape[0])
