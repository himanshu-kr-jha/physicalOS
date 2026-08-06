"""Combine the local YOLOv8 distress detector with the hosted VLM.

WHY AN ENSEMBLE BEATS EITHER ALONE
Measured on real footage, the two models fail in opposite directions:

  post_cons YOLOv8   tight, well-localised boxes, and it separates transverse
                     from longitudinal cracking. Blind to everything else in the
                     taxonomy -- the current weights know those two classes only,
                     so potholes, rutting, water and refuse all fall to the VLM.

  hosted VLM         recognises anything you can describe in words and explains
                     itself in a sentence a human can check. But on real
                     photographs its boxes are often whole-region or whole-frame,
                     which localises to nothing useful.

So: take geometry from YOLO, take coverage and reasoning from the VLM, and where
they agree treat that agreement as corroboration.

MERGE RULE
Same class and IoU >= iou_merge -> one detection. YOLO's box wins (it is the
better localiser), confidences combine, and both explanations are kept so the
evidence panel shows two independent models concurred.

Different classes are never merged even when they overlap: a pothole inside a
waterlogged stretch is two true and separate facts.

A VLM box covering most of the frame is kept only if no YOLO box supports it, and
is demoted -- it says "something is wrong along here", not "the defect is here".
"""

from __future__ import annotations

import threading
from pathlib import Path

from ..config import DomainConfig
from ..schema import BOX_SCALE, Detection, Frame


def iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def area_frac(b: list[float]) -> float:
    return ((b[2] - b[0]) / BOX_SCALE) * ((b[3] - b[1]) / BOX_SCALE)


class EnsembleDetector:
    """Runs the ONNX distress model and the hosted VLM, then merges."""

    name = "ensemble"

    def __init__(
        self,
        domain: DomainConfig,
        cache_dir: Path | None = None,
        model_path: Path | None = None,
        model: str | None = None,
        classes_per_call: int = 0,
        conf_threshold: float = 0.30,
        iou_merge: float = 0.45,
        vlm_area_cap: float = 0.35,
        vlm_optional: bool = True,
        tile: int = 0,
        intra_op_threads: int = 0,
    ):
        from .onnx_yolo import OnnxYoloDetector

        self.domain = domain
        self.iou_merge = iou_merge
        self.vlm_area_cap = vlm_area_cap

        # The local model is the backbone: free, offline, precise. Required.
        # `tile` enables sliced inference, which recovers the small distant
        # defects a single letterboxed pass loses -- see OnnxYoloDetector.
        self.yolo = OnnxYoloDetector(
            domain,
            model_path=model_path,
            conf_threshold=conf_threshold,
            tile=tile,
            intra_op_threads=intra_op_threads,
        )

        # The VLM adds the classes YOLO cannot see. Optional, because a missing
        # API key or a network outage should degrade the run, not kill it.
        self.vlm = None
        self.vlm_error: str | None = None
        try:
            from .cosmos import CosmosDetector

            self.vlm = CosmosDetector(
                domain,
                cache_dir=cache_dir,
                model=model,
                classes_per_call=classes_per_call,
            )
        except Exception as exc:  # noqa: BLE001
            self.vlm_error = str(exc).splitlines()[0]
            if not vlm_optional:
                raise

        self.merged = 0
        self.from_yolo = 0
        self.from_vlm = 0
        self.vlm_wide = 0
        # detect() runs on several frames at once under --workers, so the
        # per-frame tallies are accumulated locally and folded in once, here.
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------

    @property
    def api_calls(self) -> int:
        return getattr(self.vlm, "api_calls", 0)

    @property
    def cache_hits(self) -> int:
        return getattr(self.vlm, "cache_hits", 0)

    def probe(self) -> str:
        """Report what is actually wired up."""
        bits = [f"onnx:{self.yolo.model_path.name}"]
        if self.vlm is not None:
            bits.append(f"vlm:{self.vlm.probe()}")
        else:
            bits.append(f"vlm:UNAVAILABLE ({self.vlm_error})")
        return " + ".join(bits)

    def detect(self, frame: Frame, frame_path: Path) -> list[Detection]:
        yolo_dets = self.yolo.detect(frame, frame_path)
        vlm_dets: list[Detection] = []
        if self.vlm is not None:
            try:
                vlm_dets = self.vlm.detect(frame, frame_path)
            except Exception:  # noqa: BLE001 - one bad frame must not stop a run
                vlm_dets = []

        out = list(yolo_dets)
        n_merged = n_vlm = n_wide = 0
        used: set[int] = set()

        for v in vlm_dets:
            match, best = -1, 0.0
            for i, y in enumerate(out):
                if i in used or y.cls != v.cls:
                    continue
                s = iou(y.box, v.box)
                if s >= self.iou_merge and s > best:
                    match, best = i, s

            if match >= 0:
                # Both models agree. Keep YOLO's geometry, raise confidence, and
                # keep both explanations so a human sees it was corroborated.
                y = out[match]
                used.add(match)
                n_merged += 1
                combined = min(
                    0.99, y.confidence + (1.0 - y.confidence) * v.confidence
                )
                out[match] = y.model_copy(
                    update={
                        "confidence": round(combined, 3),
                        "severity": max(y.severity, v.severity),
                        "evidence": (
                            f"{y.evidence} Corroborated by the VLM: {v.evidence}"
                        ),
                    }
                )
                continue

            # No YOLO support. A near-frame-wide VLM box localises to nothing, so
            # keep it only as a weak area signal.
            if area_frac(v.box) > self.vlm_area_cap:
                n_wide += 1
                out.append(
                    v.model_copy(
                        update={
                            "confidence": round(min(v.confidence, 0.45), 3),
                            "evidence": (
                                f"{v.evidence} [VLM only, box covers "
                                f"{area_frac(v.box) * 100:.0f}% of the frame -- "
                                "treat as an area indication, not a point]"
                            ),
                        }
                    )
                )
            else:
                out.append(v)
            n_vlm += 1

        with self._stats_lock:
            self.from_yolo += len(yolo_dets)
            self.merged += n_merged
            self.from_vlm += n_vlm
            self.vlm_wide += n_wide

        return out
