"""Local YOLOv8 pavement-distress detector (the post_cons ONNX model).

WHY THIS MATTERS
The hosted VLM is a generalist: it recognises "there is a pothole" but its boxes
degrade badly on real photographs, and it volunteers only the obvious classes.
This model was trained specifically on pavement distress, so it gives tight boxes
and distinguishes eleven PCI distress types the VLM lumps together or ignores
entirely -- ravelling, rutting, shoving, bleeding, and edge vs longitudinal vs
transverse cracking.

It runs on CPU in well under a second per frame, costs nothing per call, and
needs no network.

CONTRACT (read from the ONNX metadata, not guessed)
  input   images   float32 [1, 3, 640, 640], RGB, 0-1, no mean/std normalisation
  output  output0  float32 [1, 15, 8400]
          rows 0-3  = cx, cy, w, h  in 640-space
          rows 4-14 = 11 class scores (YOLOv8 has no separate objectness row)
  8400 anchors = 80^2 + 40^2 + 20^2 at strides 8/16/32

LETTERBOXING
The model wants a square 640x640 but road frames are 16:9. We letterbox -- scale
to fit, pad the remainder -- rather than stretching, because stretching distorts
aspect ratio and the model was trained on letterboxed images. The pad offsets are
removed again when converting boxes back to frame coordinates.

LICENCE NOTE
These weights are an Ultralytics YOLOv8 export, and Ultralytics ships under
AGPL-3.0. Fine for academic and internal use. AGPL is copyleft and
network-triggered, so take advice before shipping it inside a closed product.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ..config import DomainConfig
from ..schema import BOX_SCALE, Detection, Frame

DEFAULT_MODEL = (
    Path.home()
    / "Documents/Cognecto/vision-stack/infrastructure/triton/models/post_cons/1/model.onnx"
)

# Mapped by INDEX, deliberately. The model's metadata spells index 4
# "longitudial crack"; we do not propagate that typo into our schema. These keys
# must match configs/domains/road_pci.yaml.
CLASS_KEYS = [
    "alligator_crack",     # 0
    "bleeding",            # 1
    "depression",          # 2
    "edge_crack",          # 3
    "longitudinal_crack",  # 4
    "patching",            # 5
    "pothole",             # 6
    "ravelling",           # 7
    "rutting",             # 8
    "shoving",             # 9
    "transverse_crack",    # 10
]

INPUT_SIZE = 640


class OnnxYoloError(RuntimeError):
    pass


def letterbox_params(w: int, h: int, size: int = INPUT_SIZE):
    """Scale and padding that fit a w x h frame into a square `size` canvas."""
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    return scale, (size - new_w) / 2.0, (size - new_h) / 2.0, new_w, new_h


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> list[int]:
    """Greedy non-maximum suppression on xyxy boxes.

    The raw model emits one prediction per anchor, so a single pothole arrives as
    a dozen near-identical boxes (measured: 13 boxes on one Kohima frame, all at
    cx=376, cy=331). Without this, every defect would be counted many times.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thresh]
    return keep


def severity_from(cls_key: str, conf: float, box_area_frac: float) -> int:
    """Map a detection to a 1-5 severity.

    The model outputs only a class and a confidence -- it was never trained to
    grade severity -- so this derives one from what we do know: how serious the
    distress TYPE is, nudged by confidence and detection size. A documented
    heuristic, not a measurement.
    """
    base = {
        "pothole": 4,
        "depression": 4,
        "rutting": 4,
        "alligator_crack": 3,
        "shoving": 3,
        "edge_crack": 3,
        "ravelling": 3,
        "longitudinal_crack": 2,
        "transverse_crack": 2,
        "bleeding": 2,
        "patching": 1,
    }.get(cls_key, 3)

    if conf >= 0.70 and box_area_frac > 0.02:
        base += 1
    elif conf < 0.40:
        base -= 1
    return max(1, min(5, base))


class OnnxYoloDetector:
    """Runs the post_cons YOLOv8 pavement-distress model on CPU."""

    name = "onnx"

    def __init__(
        self,
        domain: DomainConfig,
        model_path: Path | None = None,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
        # 0 = single letterboxed pass (the original behaviour). 640 = full frame
        # plus overlapping 640-px tiles, which recovers small distant defects the
        # 3x letterbox downscale destroys. Measured 2.83x more detections on real
        # footage, at roughly 8x the CPU time -- so it is opt-in.
        tile: int = 0,
        tile_overlap: int = 128,
        # Threads ORT may use INSIDE one session.run. 0 leaves it to ORT, which
        # grabs every core -- right when frames are processed one at a time, and
        # wrong when the caller is already running N frames concurrently, where
        # N inferences x all-cores oversubscribes and every frame gets slower.
        # The CLI sets this to 1 whenever --workers > 1.
        intra_op_threads: int = 0,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OnnxYoloError(
                "onnxruntime is required for --backend onnx:\n"
                "  uv pip install onnxruntime"
            ) from exc

        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise OnnxYoloError(
                f"Model not found at {path}\n"
                "Pass --model-path pointing at post_cons/1/model.onnx"
            )

        self.domain = domain
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.model_path = path

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            opts.intra_op_num_threads = intra_op_threads
            # One frame per thread already saturates the machine, so there is
            # nothing left for ORT to parallelise between operators.
            opts.inter_op_num_threads = 1

        # ORT sessions are safe to call from several threads at once -- that is
        # what makes the CLI's frame-level pool possible without a session per
        # worker, which would multiply the model's memory by the worker count.
        from ..ortproviders import describe, providers

        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=providers()
        )
        # Recorded rather than assumed: a CUDA wheel that fails to register looks
        # exactly like a machine with no GPU. This is the pipeline's hot spot
        # (2.5 s/frame on one CPU thread), so which device it landed on is the
        # first thing worth knowing when a run is slow.
        self.device = describe(self.session)
        self.input_name = self.session.get_inputs()[0].name

        # Only emit classes the active domain knows, so scoring and clustering
        # use real weights and radii rather than silent fallbacks.
        self.known = set(domain.class_map)
        self.skipped: set[str] = set()
        self.frames_seen = 0
        self.raw_kept = 0
        # detect() runs concurrently under --workers. `+=` on an int is a
        # read-modify-write, so without this the run summary quietly undercounts.
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------

    def _infer_raw(self, img) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the net on one PIL image. Returns boxes in THAT image's pixels.

        Split out of detect() so the same code serves a full frame and a tile.
        """
        from PIL import Image

        w0, h0 = img.size
        scale, pad_x, pad_y, new_w, new_h = letterbox_params(w0, h0)
        canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
        canvas.paste(
            img.resize((new_w, new_h), Image.BILINEAR),
            (int(round(pad_x)), int(round(pad_y))),
        )

        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        pred = self.session.run(None, {self.input_name: x})[0][0].T  # 8400 x 15

        scores = pred[:, 4 : 4 + len(CLASS_KEYS)]
        best = scores.max(axis=1)
        cls_idx = scores.argmax(axis=1)
        mask = best >= self.conf_threshold
        if not mask.any():
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

        xywh, best, cls_idx = pred[mask, :4], best[mask], cls_idx[mask]
        cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        # Undo letterbox, back into this image's own pixel frame.
        xyxy = (xyxy - np.array([pad_x, pad_y, pad_x, pad_y])) / scale
        return xyxy, best, cls_idx

    def _infer_tiled(self, img) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full frame PLUS overlapping tiles, all in frame pixels.

        WHY TILING HELPS -- measured, not assumed
        The net wants 640x640 and a road frame is 1920x1080, so letterboxing
        shrinks everything 3x. The median detected object is 108 px in-frame but
        only 36 px at the input, and 21% of detections land under 20 px, about the
        limit the model can resolve. Anything further down the road is smaller
        still, which is precisely the recall being missed.

        Tiles of 640 px need no downscaling, so a distant pothole arrives at
        native size. Measured over 20 real Kohima frames: 6 detections -> 17.

        WHY THE FULL FRAME IS KEPT TOO
        Tiles alone LOSE large objects: something spanning a tile boundary is cut
        into fragments that each fall below threshold. Measured, tiles-only went
        1 -> 0 on two frames. The union of full frame and tiles lost nothing on
        any of the 20. That union is what makes this slicing-aided HYPER
        inference rather than plain slicing.
        """
        boxes: list[np.ndarray] = []
        confs: list[np.ndarray] = []
        clss: list[np.ndarray] = []

        b, s, c = self._infer_raw(img)
        if len(b):
            boxes.append(b)
            confs.append(s)
            clss.append(c)

        w, h = img.size
        step = max(self.tile - self.tile_overlap, 64)
        for y in range(0, max(h - self.tile_overlap, 1), step):
            for x in range(0, max(w - self.tile_overlap, 1), step):
                x2, y2 = min(x + self.tile, w), min(y + self.tile, h)
                tb, ts, tc = self._infer_raw(img.crop((x, y, x2, y2)))
                if len(tb):
                    boxes.append(tb + np.array([x, y, x, y]))
                    confs.append(ts)
                    clss.append(tc)

        if not boxes:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        return np.concatenate(boxes), np.concatenate(confs), np.concatenate(clss)

    def detect(self, frame: Frame, frame_path: Path) -> list[Detection]:
        from PIL import Image

        with Image.open(frame_path) as im:
            img = im.convert("RGB")
            w0, h0 = img.size
            if self.tile and min(w0, h0) > self.tile:
                xyxy, best, cls_idx = self._infer_tiled(img)
            else:
                xyxy, best, cls_idx = self._infer_raw(img)

        with self._stats_lock:
            self.frames_seen += 1
        if not len(xyxy):
            return []

        # Class-wise NMS: overlapping boxes of DIFFERENT distress types are
        # legitimate (a pothole inside an alligator-cracked patch), so suppress
        # only within a class. This also collapses the duplicates that tiling
        # necessarily produces in the overlap regions.
        keep: list[int] = []
        for c in np.unique(cls_idx):
            sel = np.where(cls_idx == c)[0]
            keep.extend(int(sel[k]) for k in nms(xyxy[sel], best[sel], self.iou_threshold))

        keep.sort(key=lambda i: -best[i])
        keep = keep[: self.max_detections]

        out_dets: list[Detection] = []
        for i in keep:
            # Frame pixels to the canonical 0..1000 top-left convention.
            b = [
                max(0.0, min(BOX_SCALE, xyxy[i, 0] / w0 * BOX_SCALE)),
                max(0.0, min(BOX_SCALE, xyxy[i, 1] / h0 * BOX_SCALE)),
                max(0.0, min(BOX_SCALE, xyxy[i, 2] / w0 * BOX_SCALE)),
                max(0.0, min(BOX_SCALE, xyxy[i, 3] / h0 * BOX_SCALE)),
            ]
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue

            key = CLASS_KEYS[int(cls_idx[i])]
            if key not in self.known:
                self.skipped.add(key)
                continue

            conf = float(best[i])
            area_frac = ((b[2] - b[0]) / BOX_SCALE) * ((b[3] - b[1]) / BOX_SCALE)
            label = self.domain.spec(key).label

            out_dets.append(
                Detection(
                    frame_id=frame.frame_id,
                    cls=key,
                    box=b,
                    severity=severity_from(key, conf, area_frac),
                    confidence=round(conf, 3),
                    evidence=(
                        f"post_cons YOLOv8 detected {label.lower()} at "
                        f"{conf * 100:.0f}% confidence, covering "
                        f"{area_frac * 100:.1f}% of the frame."
                    ),
                )
            )

        with self._stats_lock:
            self.raw_kept += len(out_dets)
        return out_dets
