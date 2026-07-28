"""CLIP as an open-vocabulary scorer, so that what the pipeline looks for is text.

WHAT THIS IS FOR
Every other detector here has its taxonomy baked in at build time: the ONNX model
knows 11 pavement distress classes and nothing else, and the VLM is prompted from
a domain YAML shared by every run. Neither can be told "on this video, also find
loose paving slabs". CLIP can, because its class list is just sentences.

WHAT CLIP CANNOT DO, AND WHY THAT SHAPES EVERYTHING
CLIP has NO LOCALISATION. It turns one image into one vector and compares it to
text vectors. It cannot draw a box, cannot say "there, at those pixels", and
cannot count. So it is useless alone and useful as a scorer of regions something
else proposed -- an existing detection's crop, or a tile of the road mask.
Anything phrased as "CLIP detected a pothole" really means "CLIP agreed this crop
looks more like a pothole than like intact asphalt".

SIMILARITY IS RELATIVE, NEVER ABSOLUTE
The commonest way to get this wrong is to score a crop against one positive
sentence and threshold the number. CLIP will return a respectable similarity for
"a pothole in the road" against a photograph of a cat; the value only means
something COMPARED with other sentences. So every decision here is a softmax over
positives AND competing negatives, and a concept with no negatives of its own
borrows a shared background set. See pos/prompts.py.

MODEL CHOICE
Default ViT-L/14 with the OpenAI weights, on CPU, because there is no GPU here
(nvidia-smi returns nothing). Its embeddings are 768-d where ViT-B/32's are
512-d, so nothing may hardcode a width. L/14 is roughly 81 GFLOPs per 224x224
crop, a real cost on CPU: run `uv run python -m pos.clipmodel` to measure
crops/sec on this machine before choosing a tiling budget, and fall back to
ViT-B-32 when the answer is too slow.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# "-quickgelu", not plain "ViT-L-14", and this is not cosmetic. The OpenAI CLIP
# weights were trained with QuickGELU activations; open_clip 3.x pairs the plain
# arch name with standard GELU and only WARNS about the mismatch:
#   "QuickGELU mismatch between final model config (quick_gelu=False) and
#    pretrained tag 'openai' (quick_gelu=True)"
# Loading proceeds, embeddings come out the right shape, similarities look
# plausible, and every one of them is computed through the wrong non-linearity.
# A silent accuracy leak is worse than a crash, so the arch is pinned to match
# the weights.
DEFAULT_MODEL = "ViT-L-14-quickgelu"
DEFAULT_PRETRAINED = "openai"

# Crops are scored in batches because a batch amortises the per-call overhead
# that dominates one 224x224 forward pass. 16 keeps memory modest at L/14 while
# still filling the cores.
DEFAULT_BATCH = 16

# Context padding around a detection box before scoring. A pixel-tight crop of a
# pothole is a grey blob with no scale and no surroundings, and CLIP reads
# context: the same crop with a little road around it is recognisably a hole in a
# road. Fraction of the box's own width and height.
CROP_PAD_FRAC = 0.15


class ClipError(RuntimeError):
    """Raised when CLIP cannot be loaded or used, saying what to do about it."""


@dataclass(frozen=True)
class TextBank:
    """Encoded prompt text, ready to score crops against.

    `owner` maps each row of `matrix` back to the concept key that supplied it and
    `is_positive` says whether the row argues for that concept or against it. Both
    are needed because scoring is one softmax over every row at once: the winning
    row must be attributable to a concept and a polarity, or the result cannot be
    turned back into a decision.
    """

    matrix: np.ndarray  # (n_prompts, dim), L2-normalised
    texts: tuple[str, ...]
    owner: tuple[str, ...]  # concept key per row; "" for shared background
    is_positive: tuple[bool, ...]
    logit_scale: float
    model_id: str


class ClipScorer:
    """One loaded CLIP model, plus a crop-embedding cache.

    Threading: torch parallelises a batch across cores itself, so this is meant to
    be driven by ONE thread submitting batches. Handing it to a ThreadPoolExecutor
    as well would oversubscribe the machine in exactly the way measured for the
    ONNX detector -- see the intra_op_threads note in pos/perception/onnx_yolo.py.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        cache_dir: Path | None = None,
        threads: int = 0,
        device: str | None = None,
    ):
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ClipError(
                "CLIP needs torch and open_clip_torch:\n"
                "  uv add torch open_clip_torch"
            ) from exc

        self._torch = torch
        from .ortproviders import torch_device

        self.device = device or torch_device()
        # Thread count only means anything on CPU; on CUDA the work is on the
        # device and this would just cap the host-side dataloading.
        if threads > 0 and self.device == "cpu":
            torch.set_num_threads(threads)

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
        except Exception as exc:  # noqa: BLE001 - surface the real reason
            raise ClipError(
                f"could not load CLIP {model_name}/{pretrained}: {exc}\n"
                "First use downloads ~1.7 GB of weights for ViT-L-14, so this needs "
                "network. Offline, pass --clip-model ViT-B-32 for a much smaller "
                "download, or point HF_HOME at a populated cache."
            ) from exc

        model.eval()
        model.to(self.device)
        # fp16 on CUDA only. It roughly halves the time and the memory, and for
        # cosine similarity between normalised vectors the precision loss is far
        # below the margins any decision here turns on. On CPU fp16 is usually
        # SLOWER, because there is no hardware path for it.
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.dtype == torch.float16:
            model.half()
        self.model = model
        # open_clip's own transform, not a hand-rolled one: the normalisation
        # constants and the BICUBIC resize + centre-crop must match the weights
        # exactly, and reimplementing them is a silent accuracy leak.
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_id = f"{model_name}/{pretrained}"

        with torch.no_grad():
            self.logit_scale = float(model.logit_scale.exp().item())

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hits = 0
        self.encoded = 0

    # ------------------------------------------------------------------ text

    def encode_text(
        self,
        texts: list[str],
        owner: list[str],
        is_positive: list[bool],
    ) -> TextBank:
        """Encode every prompt once. This is deliberately not per-frame work.

        Encoding text costs about as much as encoding an image, so doing it inside
        the frame loop would roughly double a run for no benefit -- a prompt set is
        a handful of sentences and never changes mid-run.
        """
        if not texts:
            raise ClipError("no prompt text to encode")

        torch = self._torch
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        # float32 on the way out regardless of compute dtype: the bank is a small
        # matrix reused for every crop, and the scoring maths in score() is numpy.
        return TextBank(
            matrix=feats.float().cpu().numpy().astype(np.float32),
            texts=tuple(texts),
            owner=tuple(owner),
            is_positive=tuple(is_positive),
            logit_scale=self.logit_scale,
            model_id=self.model_id,
        )

    # ----------------------------------------------------------------- image

    def _cache_path(self, digest: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{digest}.npy"

    def _digest(self, img) -> str:
        """Key a crop by its PIXELS and by the model that will encode them.

        Hashing the raw buffer rather than a re-encoded JPEG keeps this cheap and
        exact. The model id belongs in the key because a ViT-B/32 embedding is
        512-d and a ViT-L/14 one is 768-d: serving one for the other is a shape
        error at best and silent nonsense at worst.
        """
        h = hashlib.sha256()
        h.update(self.model_id.encode())
        h.update(str(img.size).encode())
        h.update(img.tobytes())
        return h.hexdigest()[:32]

    def encode_images(self, crops: list, batch: int = DEFAULT_BATCH) -> np.ndarray:
        """Encode crops to L2-normalised vectors, in order, using the cache."""
        if not crops:
            dim = int(getattr(self.model.visual, "output_dim", 512))
            return np.zeros((0, dim), dtype=np.float32)

        torch = self._torch
        out: list[np.ndarray | None] = [None] * len(crops)
        digests: list[str | None] = []
        todo: list[int] = []

        for i, img in enumerate(crops):
            d = self._digest(img) if self.cache_dir else None
            digests.append(d)
            p = self._cache_path(d) if d else None
            if p is not None and p.exists():
                try:
                    out[i] = np.load(p)
                    self.cache_hits += 1
                    continue
                except (OSError, ValueError):
                    pass  # corrupt entry: just recompute it
            todo.append(i)

        for start in range(0, len(todo), max(1, batch)):
            idx = todo[start : start + max(1, batch)]
            tensors = torch.stack([self.preprocess(crops[i]) for i in idx])
            tensors = tensors.to(self.device, dtype=self.dtype)
            with torch.no_grad():
                feats = self.model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.float().cpu().numpy().astype(np.float32)
            self.encoded += len(idx)
            for j, i in enumerate(idx):
                out[i] = arr[j]
                p = self._cache_path(digests[i]) if digests[i] else None
                if p is not None:
                    try:
                        np.save(p, arr[j])
                    except OSError:
                        pass  # a cache that cannot be written must not fail a run

        return np.stack([v for v in out if v is not None])

    # ---------------------------------------------------------------- scoring

    def score(self, image_feats: np.ndarray, bank: TextBank) -> np.ndarray:
        """Softmax probability of every prompt, for every crop.

        Softmax over the WHOLE bank -- positives and negatives together -- is the
        point. A per-concept sigmoid would reintroduce the absolute-threshold
        mistake this module exists to avoid.
        """
        if image_feats.size == 0:
            return np.zeros((0, bank.matrix.shape[0]), dtype=np.float32)
        logits = (image_feats @ bank.matrix.T) * bank.logit_scale
        logits -= logits.max(axis=1, keepdims=True)  # stability only, not a change
        exp = np.exp(logits)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def crop_with_context(img, box_px: tuple[float, float, float, float], pad: float = CROP_PAD_FRAC):
    """Crop a box with a margin, clamped to the frame.

    `box_px` is ABSOLUTE PIXELS (x1, y1, x2, y2). Callers holding 0..1000
    normalised boxes convert with pos/svbridge.py rather than scaling here, so
    there is exactly one place in the codebase that knows that conversion.
    """
    w, h = img.size
    x1, y1, x2, y2 = box_px
    dx = (x2 - x1) * pad
    dy = (y2 - y1) * pad
    return img.crop(
        (
            max(0, int(x1 - dx)),
            max(0, int(y1 - dy)),
            min(w, int(x2 + dx)),
            min(h, int(y2 + dy)),
        )
    )


def _benchmark() -> None:
    """Measure crops/sec on this machine, to size the tiling budget.

    Run: uv run python -m pos.clipmodel     (POS_CLIP_MODEL overrides the variant)
    """
    import time

    from PIL import Image

    model_name = os.environ.get("POS_CLIP_MODEL", DEFAULT_MODEL)
    print(f"loading {model_name}/{DEFAULT_PRETRAINED} on CPU ...")
    t0 = time.perf_counter()
    scorer = ClipScorer(model_name=model_name)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s, logit_scale={scorer.logit_scale:.1f}")
    print(f"  torch threads: {scorer._torch.get_num_threads()}")

    bank = scorer.encode_text(
        ["a pothole in the road", "intact smooth asphalt"], ["pothole", ""], [True, False]
    )
    print(f"  text bank: {bank.matrix.shape}  (embedding width {bank.matrix.shape[1]})")

    rng = np.random.default_rng(0)
    crops = [
        Image.fromarray(rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)) for _ in range(32)
    ]
    scorer.cache_dir = None  # measure compute, not disk
    feats = None
    for b in (1, 8, 16, 32):
        t0 = time.perf_counter()
        feats = scorer.encode_images(crops[:b], batch=b)
        dt = time.perf_counter() - t0
        print(f"  batch {b:2d}: {dt:6.2f}s for {b:2d} crops = {b / dt:5.2f} crops/s")

    probs = scorer.score(feats, bank)
    print(f"  scored {probs.shape}, first row sums to {probs.sum(axis=1)[0]:.3f}")


if __name__ == "__main__":
    _benchmark()
