"""Road-surface segmentation: which pixels are the drivable carriageway.

WHY THIS EXISTS
Every detection this system makes is a claim about a place, and "on the road" is
the single most useful place-qualifier we have. A pothole in the carriageway is
an asset defect; the same shape in a roadside spoil heap is not. Boxes alone
cannot tell those apart, so the pipeline needs a per-frame road mask to filter
detections by zone, to estimate carriageway width, and to draw an honest
annotated video. This module is the one place that mask is produced, because a
second implementation would drift from this one and the two would disagree about
what "road" means.

NAME COLLISION, PINNED HERE
This module segments PIXELS. It has nothing to do with the `Segment` records in
<run>/segments.json, which are stretches of route along the ground. Nothing here
reads or writes that file.

CONTRACT (read from the ONNX metadata, not guessed)
  input   input    float32 [1, 3, 256, 256], RGB, 0-1, CHW, no mean/std
  output  output   float32 [1, 1, 256, 256], LOGITS -- sigmoid(x) > 0.5 is road

The 256x256 input is the whole trade-off of this model: it is fast enough to run
over every keyframe on CPU, but it resolves the road only to about 7.5 px of a
1920-wide frame. That is fine for "is this box on the road" and far too coarse
for measuring a crack. The upsample back to frame size therefore uses NEAREST,
never bilinear, so a caller never receives a half-road pixel the model never
predicted.

AN EMPTY MASK IS A NORMAL OUTCOME
Measured on runs/POC-1: coverage of 10-15% of the frame on open carriageway, and
0.0% on frame 00400, a crowded market scene where no drivable surface is visible.
The model is right to return nothing there. `polygon()` returns None in that
case and every caller must handle None rather than treating it as an error.

WHAT THIS MODEL IS NOT
It segments the carriageway only -- verge, kerb, footpath, debris piles and
vehicles are all excluded (verified visually on POC-1 keyframes). There is NO
footpath segmentation model in this environment. The sibling cs_seg_unet is not
one: its three channels are background / patchy-unpaved / noise. Anything
"beside the road" must be derived geometrically -- see off_carriageway_band --
and reported as a derivation, not as a segmentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np

# cv2 arrives as a hard dependency of supervision, which pyproject already
# requires; there is no separate opencv pin to keep in step.
import cv2
from PIL import Image

from .ortproviders import describe as ort_describe
from .ortproviders import providers as ort_providers

DEFAULT_MODEL = (
    Path.home()
    / "Documents/Cognecto/vision-stack/infrastructure/triton/models/road_seg_unet/1/model.onnx"
)

MODEL_ENV_VAR = "POS_ROAD_SEG"

INPUT_SIZE = 256

# A component smaller than this fraction of the frame is not a carriageway, it is
# a speckle of tarmac glimpsed between people or under a vehicle. Turning those
# into a PolygonZone yields a zone that catches nothing while looking meaningful
# in the viewer, so they are dropped. 0.5% of 1920x1080 is ~10k px, while the
# real carriageway components measured on POC-1 are 10-15% of the frame -- more
# than an order of magnitude above the floor, so this rejects speckle without
# ever rejecting a road.
MIN_AREA_FRAC = 0.005


@dataclass(frozen=True)
class HorizonFit:
    """A horizon row measured from the road itself, and how much to trust it."""

    cy: float  # row where the carriageway converges; may fall outside the frame
    n_frames: int  # frames that produced an accepted fit
    n_rejected: int
    spread_px: float  # p10..p90 of the per-frame estimates: the real uncertainty
    pitch_offset_frac: float  # (cy - H/2) / H, the form CameraConfig wants
    m_per_px_at_bottom: float  # metres per pixel on the bottom row, for sanity


def _fit_one(mask: np.ndarray) -> tuple[float, float] | None:
    """Horizon row from one frame, by extrapolating the road to zero width.

    THE IDEA, AND WHY IT NEEDS NEITHER GPS NOR MARKERS
    On flat ground a carriageway of constant real width W images with pixel width
        px(v) = W * (v - cy) / h
    which is LINEAR in the row v and reaches zero at the horizon. Fit a line to
    the mask's width per row and the x-intercept IS cy. Nothing about motion,
    speed or focal length enters -- which is precisely why this succeeds where
    the motion fit fails: f cancels out of width entirely, leaving one unknown
    that every row of the mask constrains at once.
    """
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) < 80:
        return None
    lo, hi = int(rows[0]), int(rows[-1])

    # Skip the top 30% of the road: approaching the vanishing point the mask is
    # a handful of noisy pixels, and the linear model is swamped by them.
    band = np.linspace(lo + 0.30 * (hi - lo), hi, 40).astype(int)
    vs: list[int] = []
    ws: list[int] = []
    for v in band:
        xs = np.where(mask[v])[0]
        if len(xs) < 2:
            continue
        # A row touching a frame edge is a LOWER BOUND on width, not a width,
        # and including it bends the line towards a false horizon.
        if xs[0] == 0 or xs[-1] == mask.shape[1] - 1:
            continue
        vs.append(int(v))
        ws.append(int(xs[-1] - xs[0] + 1))

    if len(vs) < 10:
        return None

    slope, intercept = np.polyfit(vs, ws, 1)
    if slope <= 1e-6:
        return None  # width not growing downwards: not a road receding from us

    w_arr = np.asarray(ws, dtype=float)
    pred = slope * np.asarray(vs, dtype=float) + intercept
    ss_res = float(np.sum((w_arr - pred) ** 2))
    ss_tot = float(np.sum((w_arr - w_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-intercept / slope), r2


def estimate_horizon(
    masks: Sequence[np.ndarray],
    frame_h: int,
    height_m: float = 1.35,
    min_r2: float = 0.80,
) -> HorizonFit | None:
    """Measure the horizon row across many frames, robustly.

    One frame is not enough. A bend, a junction, a side road the segmenter joined
    onto, or a crest all break the straight-flat-road assumption, and on real
    footage that is common rather than rare -- measured on runs/POC-1, single
    frames ranged from cy = -801 to +854 while the median sat at 678. So each
    frame is fitted alone, poor fits are dropped on R-squared and on landing
    outside the band a horizon could occupy, and the survivors are combined by
    MEDIAN, which ignores the outliers a mean would swallow.

    `spread_px` is the p10..p90 of the survivors and is the honest uncertainty: a
    wide spread means the road was not behaving like a straight flat ribbon, and
    the caller should treat the answer as weaker rather than pretend otherwise.
    """
    fits: list[float] = []
    rejected = 0
    for m in masks:
        got = _fit_one(m)
        if got is None:
            rejected += 1
            continue
        cy, r2 = got
        # A horizon has to be somewhere a horizon could be. A camera aimed
        # steeply down can put it above the frame, so the upper bound is
        # generous, but a horizon below the road it was measured from is
        # arithmetically impossible.
        if r2 < min_r2 or not (-0.5 * frame_h <= cy <= 0.95 * frame_h):
            rejected += 1
            continue
        fits.append(cy)

    if len(fits) < 5:
        return None

    fits.sort()
    cy = float(median(fits))
    spread = float(fits[int(len(fits) * 0.9)] - fits[int(len(fits) * 0.1)])
    bottom = frame_h - 1
    return HorizonFit(
        cy=cy,
        n_frames=len(fits),
        n_rejected=rejected,
        spread_px=spread,
        pitch_offset_frac=(cy - frame_h / 2.0) / frame_h,
        m_per_px_at_bottom=height_m / (bottom - cy) if bottom > cy else float("nan"),
    )


class RoadSegError(RuntimeError):
    pass


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Explicit argument, then POS_ROAD_SEG, then the known triton path.

    The env var sits in the middle deliberately: it lets an operator relocate the
    weights without touching code or every call site, while an explicit argument
    still wins so a test can pin one specific file.
    """
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path).expanduser())
    env = os.environ.get(MODEL_ENV_VAR, "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(DEFAULT_MODEL)

    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  {p}" for p in candidates)
    raise RoadSegError(
        "road_seg_unet ONNX model not found. Tried:\n"
        f"{tried}\n"
        f"Set {MODEL_ENV_VAR} to the path of road_seg_unet/1/model.onnx, "
        "or pass model_path=."
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # The net emits logits and they reach around +-30 on confident pixels, where a
    # naive exp(-x) overflows and prints a RuntimeWarning per frame. Clipping
    # first is exact at float32 precision here: sigmoid(-30) is 9e-14.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class RoadSegmenter:
    """Runs road_seg_unet on CPU and hands back frame-resolution road masks."""

    name = "road_seg_unet"

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        min_area_frac: float = MIN_AREA_FRAC,
        # Threads ORT may use INSIDE one session.run. 0 leaves it to ORT, which
        # grabs every core -- right when frames are processed one at a time, and
        # wrong when the caller is already running N frames concurrently, where N
        # inferences x all-cores oversubscribes and every frame gets slower. Any
        # stage following the --workers pattern in `pos perceive` should pass 1.
        intra_op_threads: int = 0,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RoadSegError(
                "onnxruntime is required for road segmentation:\n"
                "  uv pip install onnxruntime"
            ) from exc

        path = resolve_model_path(model_path)
        self.model_path = path
        self.threshold = threshold
        self.min_area_frac = min_area_frac

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            opts.intra_op_num_threads = intra_op_threads
            # One frame per thread already saturates the machine, so there is
            # nothing left for ORT to parallelise between operators.
            opts.inter_op_num_threads = 1

        # One session, shared. ORT sessions are safe to call from several threads
        # at once, and this file is 93 MB of weights -- a session per worker would
        # multiply that by the worker count for no gain.
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=ort_providers()
        )
        self.device = ort_describe(self.session)
        self.input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------

    def mask(
        self,
        source: str | Path | Image.Image,
        out_wh: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Road mask for one frame, as bool (H, W) at the ORIGINAL resolution.

        `out_wh` is (WIDTH, HEIGHT) -- PIL's ordering, and the ordering of the
        width/height fields in frames.json -- while the returned array is indexed
        (row, col) = (y, x) like every other image array here. Swapping the two on
        a 1920x1080 frame yields a silently transposed mask that still looks like
        a mask, so pass frames.json's width and height straight through. None
        means "use the source image's own size".
        """
        if isinstance(source, Image.Image):
            img = source.convert("RGB")
            src_wh = img.size
        else:
            with Image.open(source) as im:
                img = im.convert("RGB")
                src_wh = img.size

        w, h = out_wh if out_wh else src_wh

        # Straight resize, no letterbox. Unlike the YOLO detector, this net was
        # trained on whole road scenes squashed to square, so preserving aspect
        # ratio here would paste grey bars where it expects sky and tarmac.
        small = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        # HWC uint8 -> CHW float32, batch of 1. RGB, not BGR: this array comes
        # from PIL so it is already RGB and must NOT be channel-reversed the way a
        # cv2.imread result would have to be.
        x = np.asarray(small, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

        logits = self.session.run(None, {self.input_name: x})[0]
        # [1, 1, 256, 256] -> [256, 256]; the single channel IS road-vs-not.
        # These are logits, so the sigmoid is mandatory -- with threshold 0.5 it
        # happens to reduce to logit > 0, but it is kept explicit so that raising
        # threshold to 0.7 means what a caller expects.
        prob = _sigmoid(logits[0, 0])
        small_mask = (prob > self.threshold).astype(np.uint8)

        # NEAREST on the way up. Bilinear would interpolate a 0.5 band along every
        # road edge, and thresholding that band invents road pixels the model
        # never predicted -- exactly the error that leaks roadside debris into an
        # "on carriageway" zone.
        up = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return up.astype(bool)

    def coverage(self, mask: np.ndarray) -> float:
        """Fraction of the frame that is road, 0.0-1.0."""
        if mask is None or mask.size == 0:
            return 0.0
        return float(np.count_nonzero(mask)) / float(mask.size)

    def polygon(
        self,
        mask: np.ndarray,
        epsilon_frac: float = 0.01,
    ) -> np.ndarray | None:
        """Largest road component as an (N, 2) int pixel polygon, or None.

        Shaped for sv.PolygonZone, which wants ABSOLUTE PIXELS in (x, y) order --
        NOT the 0..1000 normalised boxes of pos/schema.py. Bridging those two
        conventions is the caller's job; this is the pixel side of it.

        Returns None when the mask is empty or the largest component is under the
        area floor. That is a normal outcome, not a failure: measured 0.0%
        coverage on POC-1 frame 00400, a market scene with no visible
        carriageway. Callers must branch on None rather than assume a zone.

        Only the largest component is returned. A frame often shows the
        carriageway plus slivers of it past a vehicle or a pedestrian; joining
        those into one ring produces a self-intersecting polygon whose
        point-in-polygon test answers incorrectly in the crossing region.

        THE POLYGON IS AN APPROXIMATION, THE MASK IS THE TRUTH. Measured over
        five POC-1 keyframes, the filled polygon agrees with the mask at IoU
        0.92-0.95, and it errs towards over-covering: a straight edge cuts across
        a concave kerb line and takes a little verge with it. Use it for zone
        membership, where a few pixels of slack is harmless. Do NOT use it to
        measure carriageway area or width -- for that, count mask pixels.
        """
        if mask is None or mask.size == 0 or not mask.any():
            return None

        # RETR_EXTERNAL: a hole in the road (a vehicle standing on it) is not a
        # separate road, and a ring with holes cannot be one polygon anyway. The
        # hole is absorbed, which is the safe direction -- it over-covers rather
        # than dropping carriageway.
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area_frac * mask.size:
            return None

        # epsilon as a fraction of the perimeter: the raw contour of a
        # NEAREST-upsampled 256x256 mask is a staircase of thousands of ~7-px
        # steps, slow to test points against and visibly wrong when drawn. 1% of
        # the perimeter collapses it to tens of vertices while keeping the
        # carriageway's real trapezoid.
        eps = epsilon_frac * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2)

        # A 1-2 vertex result has no interior. PolygonZone would happily build a
        # zone from it that matches nothing, and downstream that reads as "no
        # detections on the road" instead of "no road found".
        if len(approx) < 3:
            return None
        return approx.astype(int)

    def off_carriageway_band(self, mask: np.ndarray, width_px: int = 40) -> np.ndarray:
        """The strip of `width_px` just OUTSIDE the road boundary, as bool (H, W).

        HONEST DESCRIPTION OF WHAT THIS IS
        A geometric derivation of "beside the road": the road mask dilated by
        width_px, minus the road itself. It is NOT a trained footpath segmenter,
        and it cannot tell a footpath from a verge, a drain, a kerb, a shopfront
        or a parked car. No footpath segmentation model exists in this
        environment -- cs_seg_unet is background / patchy-unpaved / noise, not
        road plus footpath -- so anything reported from this band must be
        labelled "adjacent to the carriageway" and never "on the footpath".

        TWO CAVEATS THAT MATTER AT THE CALL SITE
        1. The band wraps the WHOLE boundary, including the far end at the horizon
           and the near end at the frame's bottom edge, neither of which is
           beside anything. Intersect with your own row range if you want only
           the lateral verge.
        2. width_px is pixels and pixels are not metres: 40 px at the bottom of a
           1080-tall frame is roughly a metre of ground, while near the horizon
           the same 40 px is tens of metres. Use pos/geo.py if you need a
           ground-plane width.
        """
        if mask is None or mask.size == 0 or not mask.any() or width_px <= 0:
            return np.zeros(getattr(mask, "shape", (0, 0)), dtype=bool)

        base = mask.astype(np.uint8)
        # Ellipse rather than a rect kernel so the band keeps a constant width
        # around diagonal road edges; a rect widens those by sqrt(2).
        k = 2 * int(width_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        grown = cv2.dilate(base, kernel, iterations=1).astype(bool)
        return grown & ~mask.astype(bool)
