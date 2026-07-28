"""Convert lingbot-map's NPZ predictions into a georeferenced, web-ready PLY.

lingbot-map reconstructs real 3D geometry from the video itself, which is the
most striking thing in the demo -- but it needs a CUDA GPU, so it runs ONCE
offline on a rented box (see scripts/lingbot_gpu_pass.sh) and the result is
committed as a cached asset. Nothing at runtime depends on a GPU.

Two problems this module solves:

1. lingbot-map, like any monocular reconstruction, produces geometry at an
   arbitrary scale and orientation. We fit its camera trajectory to the GPS
   track with a similarity transform (scale, yaw, translation) so the cloud
   lands in the same local ENU metres as everything else in the viewer.

2. A raw reconstruction is tens of millions of points. We voxel-downsample to a
   budget a browser can actually draw.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from .geo import latlon_to_local_m
from .schema import Frame

# Candidate key names, because they differ between lingbot-map checkpoints.
_POINT_KEYS = ("world_points", "points", "xyz", "pts3d", "point_map")
_COLOR_KEYS = ("rgb", "colors", "color", "image", "images")
_CONF_KEYS = ("conf", "confidence", "conf_map")
_POSE_KEYS = ("camera_pose", "cam2world", "extrinsic", "extrinsics", "pose")


def _first(npz, keys: tuple[str, ...]):
    for k in keys:
        if k in npz:
            return npz[k]
    return None


def load_npz_dir(pred_dir: Path, conf_threshold: float = 1.5):
    """Concatenate per-frame NPZ predictions into (points Nx3, colors Nx3, cams Mx3)."""
    files = sorted(Path(pred_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz prediction files in {pred_dir}")

    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []
    cams: list[np.ndarray] = []

    for f in files:
        with np.load(f, allow_pickle=False) as npz:
            pts = _first(npz, _POINT_KEYS)
            if pts is None:
                continue
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)

            col = _first(npz, _COLOR_KEYS)
            if col is not None:
                col = np.asarray(col).reshape(-1, 3)
                if col.shape[0] != pts.shape[0]:
                    col = None
            if col is None:
                col = np.full_like(pts, 160.0, dtype=np.float32)
            col = np.asarray(col, dtype=np.float32)
            # Colours arrive as either 0..1 floats or 0..255 ints.
            if float(col.max()) <= 1.001:
                col = col * 255.0

            conf = _first(npz, _CONF_KEYS)
            if conf is not None:
                conf = np.asarray(conf, dtype=np.float32).reshape(-1)
                if conf.shape[0] == pts.shape[0]:
                    keep = conf >= conf_threshold
                    pts, col = pts[keep], col[keep]

            finite = np.isfinite(pts).all(axis=1)
            pts_all.append(pts[finite])
            col_all.append(col[finite])

            pose = _first(npz, _POSE_KEYS)
            if pose is not None:
                pose = np.asarray(pose, dtype=np.float32)
                if pose.shape[-2:] == (4, 4):
                    cams.append(pose.reshape(-1, 4, 4)[-1][:3, 3])
                elif pose.size >= 3:
                    cams.append(pose.reshape(-1)[:3])

    if not pts_all:
        raise ValueError(f"No usable point arrays found in {pred_dir}")

    return (
        np.concatenate(pts_all, axis=0),
        np.concatenate(col_all, axis=0),
        np.asarray(cams, dtype=np.float32) if cams else np.zeros((0, 3), np.float32),
    )


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, budget: int = 400_000
) -> tuple[np.ndarray, np.ndarray]:
    """Keep roughly `budget` points by snapping to a voxel grid and deduping.

    Voxel-based rather than random: random sampling thins dense, well-observed
    surfaces and sparse noise equally, so the road ends up speckled while the
    junk survives. Voxels keep one point per occupied cell, preserving
    structure at a uniform density.
    """
    n = points.shape[0]
    if n <= budget:
        return points, colors

    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = float(np.linalg.norm(hi - lo)) or 1.0

    def occupied(voxel: float) -> np.ndarray:
        keys = np.floor((points - lo) / max(voxel, 1e-6)).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return idx

    # Bisect on voxel size rather than only growing it. A road cloud is
    # essentially a 2D SURFACE, not a filled volume, so a size estimated from
    # extent/budget^(1/3) is far too coarse -- measured: it returned 9,974 points
    # against a 500,000 budget. Searching both directions gets close to budget
    # whatever the cloud's effective dimensionality.
    small, large = extent / 4000.0, extent
    best = occupied(large)

    for _ in range(30):
        mid = (small + large) / 2.0
        idx = occupied(mid)
        if idx.size > budget:
            small = mid          # too many points: coarsen
        else:
            large = mid          # fits: try finer
            best = idx
            if idx.size > budget * 0.85:
                break            # close enough, stop bisecting
        if (large - small) / max(large, 1e-9) < 0.01:
            break

    if best.size == 0:
        best = np.arange(min(n, budget))
    idx = np.sort(best)
    return points[idx], colors[idx]


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Best-fit scale, yaw and translation mapping src onto dst in the XZ plane.

    Only yaw is solved for. The vehicle path is essentially planar, and letting
    roll and pitch float on a noisy monocular trajectory tilts the whole street.
    Returns (scale, yaw_radians, translation_xz).
    """
    if src.shape[0] < 2 or dst.shape[0] < 2:
        return 1.0, 0.0, np.zeros(2, np.float32)

    n = min(src.shape[0], dst.shape[0])
    a = src[:n][:, [0, 2]].astype(np.float64)
    b = dst[:n].astype(np.float64)

    ca, cb = a.mean(axis=0), b.mean(axis=0)
    a0, b0 = a - ca, b - cb

    norm_a = float(np.sqrt((a0**2).sum()))
    scale = float(np.sqrt((b0**2).sum()) / norm_a) if norm_a > 1e-9 else 1.0

    # Kabsch in 2D, reduced to a single angle.
    num = float((a0[:, 0] * b0[:, 1] - a0[:, 1] * b0[:, 0]).sum())
    den = float((a0[:, 0] * b0[:, 0] + a0[:, 1] * b0[:, 1]).sum())
    yaw = math.atan2(num, den)

    c, s = math.cos(yaw), math.sin(yaw)
    rotated = np.stack(
        [scale * (c * a[:, 0] - s * a[:, 1]), scale * (s * a[:, 0] + c * a[:, 1])],
        axis=1,
    )
    trans = (b - rotated).mean(axis=0)
    return scale, yaw, trans.astype(np.float32)


def write_ply(
    points: np.ndarray,
    colors: np.ndarray,
    out_path: Path,
    comment: str = "Generated by PhysicalOS",
) -> None:
    """Binary little-endian PLY with per-vertex colour.

    `comment` records HOW the cloud was made. Two very different pipelines write
    these files -- a lingbot-map GPU reconstruction and CPU monocular depth --
    and the header is the only place that provenance survives.
    """
    n = int(points.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment {comment}\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()

    xyz = np.ascontiguousarray(points[:, :3], dtype="<f4")
    rgb = np.ascontiguousarray(np.clip(colors[:, :3], 0, 255), dtype=np.uint8)

    with open(out_path, "wb") as fh:
        fh.write(header)
        # Interleave via one structured array: a per-point struct.pack loop
        # takes minutes on a 400k cloud.
        rec = np.empty(
            n,
            dtype=np.dtype(
                [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                 ("r", "u1"), ("g", "u1"), ("b", "u1")]
            ),
        )
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        fh.write(rec.tobytes())


def build_pointcloud(
    pred_dir: Path,
    frames: list[Frame],
    out_path: Path,
    origin: tuple[float, float],
    budget: int = 400_000,
    conf_threshold: float = 1.5,
    georeference: bool = True,
) -> int:
    """Full conversion. Returns the number of points written."""
    points, colors, cams = load_npz_dir(pred_dir, conf_threshold=conf_threshold)

    if georeference and cams.shape[0] >= 2 and len(frames) >= 2:
        # Match the reconstructed trajectory against the GPS track in local ENU.
        track = np.array(
            [latlon_to_local_m(f.lat, f.lon, origin[0], origin[1]) for f in frames],
            dtype=np.float64,
        )
        # latlon_to_local_m returns (east, north); compare against (x, z).
        idx = np.linspace(0, len(track) - 1, cams.shape[0]).round().astype(int)
        scale, yaw, trans = fit_similarity(cams, track[idx])

        c, s = math.cos(yaw), math.sin(yaw)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        points = np.stack(
            [
                scale * (c * x - s * z) + trans[0],
                scale * y,
                # store.ts uses z = -north, while trans[1] is north.
                -(scale * (s * x + c * z) + trans[1]),
            ],
            axis=1,
        ).astype(np.float32)

    points, colors = voxel_downsample(points, colors, budget=budget)
    write_ply(points, colors, Path(out_path),
              comment="PhysicalOS: lingbot-map reconstruction, georeferenced to GPS")
    return int(points.shape[0])
