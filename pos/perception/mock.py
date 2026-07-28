"""Offline detector. No GPU, no API key, no network.

Two jobs:

1. Replay ground truth produced by `scripts/make_sample.py`. The sample video
   is rendered with defects at known pixel positions, so the boxes this
   returns sit on genuinely visible things. That makes the evidence panel
   honest even in the offline demo -- you can see the pothole under the box.

2. On an arbitrary video with no ground truth, emit deterministic pseudo-
   detections so the pipeline and viewer can still be exercised end to end.

Mode 2 is clearly synthetic and every `evidence` string says so, to make sure
nobody mistakes it for real perception.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import DomainConfig
from ..schema import Detection, Frame


class MockDetector:
    """Replays fixtures, or fabricates deterministic ones as a fallback."""

    name = "mock"

    def __init__(
        self,
        domain: DomainConfig,
        truth_path: Path | None = None,
        density: float = 0.45,
    ):
        self.domain = domain
        self.density = density
        self.truth: dict[str, list[dict]] = {}

        if truth_path and Path(truth_path).exists():
            try:
                self.truth = json.loads(Path(truth_path).read_text())
            except json.JSONDecodeError:
                self.truth = {}

        self.has_truth = bool(self.truth)
        # Only fabricate classes that can be placed on the ground plane.
        self._point_classes = [
            c for c in domain.classes if c.geometry == "point" and c.weight > 0
        ] or list(domain.classes)

    def detect(self, frame: Frame, frame_path: Path) -> list[Detection]:
        if self.has_truth:
            return self._from_truth(frame)
        return self._fabricate(frame)

    # ------------------------------------------------------------------ truth

    def _from_truth(self, frame: Frame) -> list[Detection]:
        """Look fixtures up by TIMESTAMP, not by frame_id.

        Frame ids are just ffmpeg's output counter; which source frame each one
        holds depends on the sampling filter and the source frame rate. Keying
        on the frame's true t_sec means the fixture lines up with the pixels
        regardless of how the video was sampled.
        """
        valid = set(self.domain.class_map)
        out: list[Detection] = []
        for item in self.truth.get(f"{frame.t_sec:.2f}", []):
            cls = item.get("cls")
            if cls not in valid:
                continue
            out.append(
                Detection(
                    frame_id=frame.frame_id,
                    cls=cls,
                    box=[float(v) for v in item["box"]],
                    severity=int(item.get("severity", 3)),
                    confidence=float(item.get("confidence", 0.9)),
                    evidence=item.get("evidence", ""),
                )
            )
        return out

    # -------------------------------------------------------------- fabricate

    def _fabricate(self, frame: Frame) -> list[Detection]:
        """Deterministic per-frame output, seeded so reruns are identical."""
        rng = random.Random(f"{self.domain.key}:{frame.frame_id}")
        if rng.random() > self.density:
            return []

        out: list[Detection] = []
        for _ in range(rng.randint(1, 2)):
            spec = rng.choice(self._point_classes)

            # Place the box in the lower half of the frame so it projects onto
            # the ground plane instead of being discarded above the horizon.
            w = rng.uniform(45, 130)
            h = rng.uniform(35, 90)
            x1 = rng.uniform(180, 1000 - 180 - w)
            y1 = rng.uniform(540, 1000 - h - 30)

            out.append(
                Detection(
                    frame_id=frame.frame_id,
                    cls=spec.key,
                    box=[round(x1), round(y1), round(x1 + w), round(y1 + h)],
                    severity=rng.randint(2, 5),
                    confidence=round(rng.uniform(0.55, 0.95), 2),
                    evidence=(
                        f"[SYNTHETIC FIXTURE] Placeholder {spec.label.lower()} for "
                        "pipeline testing. Not a real observation."
                    ),
                )
            )
        return out
