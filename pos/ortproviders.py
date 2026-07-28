"""One place that decides which onnxruntime execution providers to use.

WHY THIS EXISTS
Three separate models run through onnxruntime here -- the YOLOv8 distress detector
(pos/perception/onnx_yolo.py), the road-segmentation UNet (pos/segment.py) and the
monocular depth model (pos/depthcloud.py) -- and each named
["CPUExecutionProvider"] literally. On a GPU machine that is the worst kind of
bug: nothing errors, nothing warns, a CUDA build of onnxruntime sits there unused
and the run is merely slow. The cost of getting it wrong is large -- the YOLO pass
alone measured 2.5 s per frame single-threaded on CPU.

WHY A FALLBACK LIST RATHER THAN A CHOICE
onnxruntime takes providers in priority order and silently skips any it cannot
register. So ["CUDAExecutionProvider", "CPUExecutionProvider"] means "GPU if the
wheel and driver allow, CPU otherwise", which is what code that must run on both a
laptop and a GPU box wants. That silence is also the trap: a missing CUDA wheel is
indistinguishable from a machine with no GPU, which is why describe() reports what
was ACTUALLY registered rather than what was requested.

POS_DEVICE
  unset / "auto"  prefer CUDA, fall back to CPU
  "cuda"          expect CUDA, and complain audibly when it is not there
  "cpu"           force CPU, useful for benchmarking against the GPU path
"""

from __future__ import annotations

import os

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"


def wanted_device() -> str:
    """What the operator asked for: "auto", "cuda" or "cpu"."""
    value = os.environ.get("POS_DEVICE", "auto").strip().lower()
    return value if value in ("auto", "cuda", "cpu") else "auto"


def available() -> list[str]:
    """Providers this onnxruntime build can actually offer."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - never let a probe break a run
        return [CPU]


def providers() -> list[str]:
    """The provider list to hand to ort.InferenceSession."""
    device = wanted_device()
    if device == "cpu":
        return [CPU]

    have = available()
    if CUDA in have:
        return [CUDA, CPU]

    if device == "cuda":
        # Asked for explicitly and absent. Do not fail the run -- a slow answer
        # beats no answer -- but do not let it pass unremarked either, because the
        # usual cause is `onnxruntime` installed where `onnxruntime-gpu` was
        # meant, and that is otherwise invisible.
        print(
            "  POS_DEVICE=cuda but CUDAExecutionProvider is unavailable "
            f"(this build offers: {', '.join(have)}). Falling back to CPU. "
            "Install onnxruntime-gpu, and check nvidia-smi."
        )
    return [CPU]


def on_gpu() -> bool:
    """True when the GPU provider will actually be used."""
    return providers()[0] == CUDA


def describe(session=None) -> str:
    """One-line report of what is really in use, for a run log.

    Prefers the live session, because the authority on what got registered is the
    session itself, not the request that created it.
    """
    if session is not None:
        try:
            got = list(session.get_providers())
            return got[0] if got else "unknown"
        except Exception:  # noqa: BLE001
            pass
    return providers()[0]


def torch_device() -> str:
    """Device string for the torch-based models (CLIP). Same policy as above."""
    device = wanted_device()
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    if device == "cuda":
        print(
            "  POS_DEVICE=cuda but torch.cuda.is_available() is False. Falling "
            "back to CPU. The usual cause is a CPU-only torch wheel: check that "
            "[tool.uv.sources] in pyproject.toml points at a CUDA index."
        )
    return "cpu"
