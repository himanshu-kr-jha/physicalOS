"""Perception backends.

Every backend satisfies the same `Detector` protocol, so the pipeline never
knows or cares whether a finding came from a hosted VLM, a local GPU model,
or an offline fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import DomainConfig
from .base import Detector, build_prompt, parse_detections

__all__ = [
    "Detector",
    "build_prompt",
    "parse_detections",
    "get_detector",
    "default_workers",
    "BACKENDS",
]

BACKENDS = ("mock", "cosmos", "onnx", "ensemble", "locate-anything")


def default_workers(backend: str) -> int:
    """How many frames to have in flight at once, when the caller says "auto".

    The right number depends on where the time actually goes:

      cosmos    a network round trip per frame; the CPU sits idle almost the
                whole time, so concurrency is close to a linear speedup and the
                limit is the endpoint's tolerance, not this machine.
      ensemble  a local pass AND a network call per frame. The network still
                dominates, but each frame now costs real CPU too.
      onnx      pure CPU. More workers than cores only adds contention; one core
                is left free so the box stays usable.
      mock      microseconds per frame. A pool would cost more than it saves.
    """
    key = backend.strip().lower()
    if key == "mock":
        return 1
    if key == "cosmos":
        return 8
    if key == "ensemble":
        return 6
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def get_detector(
    name: str,
    domain: DomainConfig,
    cache_dir: Path | None = None,
    truth_path: Path | None = None,
    model: str | None = None,
    classes_per_call: int = 0,
    model_path: Path | None = None,
    tile: int = 0,
    intra_op_threads: int = 0,
) -> Detector:
    """Construct a detector by name.

    `mock` is the default everywhere so a fresh clone runs with no API key.
    """
    key = name.strip().lower()

    if key == "mock":
        from .mock import MockDetector

        return MockDetector(domain, truth_path=truth_path)

    if key == "cosmos":
        from .cosmos import CosmosDetector

        return CosmosDetector(
            domain,
            cache_dir=cache_dir,
            model=model,
            classes_per_call=classes_per_call,
        )

    if key == "onnx":
        from .onnx_yolo import OnnxYoloDetector

        return OnnxYoloDetector(
            domain,
            model_path=model_path,
            tile=tile,
            intra_op_threads=intra_op_threads,
        )

    if key == "ensemble":
        from .ensemble import EnsembleDetector

        return EnsembleDetector(
            domain,
            cache_dir=cache_dir,
            model_path=model_path,
            model=model,
            classes_per_call=classes_per_call,
            tile=tile,
        )

    if key in ("locate-anything", "locate_anything"):
        from .locate_anything import LocateAnythingDetector

        return LocateAnythingDetector(domain)

    raise ValueError(f"Unknown backend {name!r}. Choose from: {', '.join(BACKENDS)}")
