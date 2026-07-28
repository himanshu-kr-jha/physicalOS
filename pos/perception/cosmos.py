"""Cosmos Reason via the hosted NVIDIA NIM endpoint.

This is the only perception backend that needs no local GPU, so it carries the
runtime work. The endpoint is OpenAI-compatible, so the official `openai` SDK
works against it unchanged.

Two things here are defensive on purpose:

1. Every published Cosmos Reason example targets a LOCAL NIM at
   127.0.0.1:8000, so hosted `video_url` support is unverified. This backend
   therefore works per-frame, which any VLM endpoint supports.

2. Responses are cached on disk keyed by (image bytes + prompt + model). A
   300-frame run costs real money and real minutes; re-running after a crash,
   or after tweaking only the clustering step, should cost nothing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

from ..config import DomainConfig
from ..schema import Detection, Frame
from .base import build_prompt, parse_detections

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Chosen by measurement, not by reputation. Each candidate was run against the
# synthetic sample and its boxes compared with known ground truth:
#
#   nvidia/llama-3.1-nemotron-nano-vl-8b-v1  boxes within a few units of truth.
#                                            Conservative: ~1 detection/frame.
#   nvidia/nemotron-nano-12b-v2-vl           finds more, but hallucinates boxes
#                                            (e.g. whole-frame [0,0,1000,1000]).
#   meta/llama-3.2-90b-vision-instruct       obeys the JSON shape but emits
#                                            degenerate [0,0,0,0] boxes.
#   meta/llama-3.2-11b-vision-instruct       ignores the JSON contract, prose.
#
# Note that models.list() advertises models an account may not be entitled to
# invoke: cosmos-reason2-8b, vila and neva-22b all appear in the listing but
# return 404 on call. That is exactly why this fallback chain and probe() exist.
DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

MODEL_FALLBACKS = [
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "nvidia/cosmos-reason2-8b",  # preferred where an account has access
    "nvidia/nemotron-nano-12b-v2-vl",
    "meta/llama-3.2-90b-vision-instruct",
]


class CosmosError(RuntimeError):
    pass


def _b64_data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


class CosmosDetector:
    """Per-frame detection against a hosted, OpenAI-compatible VLM endpoint."""

    name = "cosmos"

    def __init__(
        self,
        domain: DomainConfig,
        cache_dir: Path | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        max_retries: int = 4,
        classes_per_call: int = 0,
    ):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not api_key:
            raise CosmosError(
                "NVIDIA_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://build.nvidia.com -- or run with `--backend mock`."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise CosmosError("The `openai` package is required: uv sync") from exc

        self.domain = domain
        self.model = model or os.environ.get("POS_MODEL", DEFAULT_MODEL)

        # classes_per_call = 0 means one open-ended pass over the whole
        # taxonomy: cheapest, but recall suffers because the model volunteers
        # only the obvious classes. Splitting the taxonomy into small batches
        # costs one API call per batch per frame and finds materially more.
        keys = [c.key for c in domain.classes]
        if classes_per_call and classes_per_call > 0:
            self.prompts = [
                build_prompt(domain, keys[i : i + classes_per_call])
                for i in range(0, len(keys), classes_per_call)
            ]
        else:
            self.prompts = [build_prompt(domain)]
        self.classes_per_call = classes_per_call

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI(
            base_url=os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            api_key=api_key,
            timeout=180.0,
        )
        self._model_confirmed = False
        self.cache_hits = 0
        self.api_calls = 0
        # This detector is the main beneficiary of --workers: each frame is a
        # network round trip, so N in flight is close to an N-times speedup.
        # `+=` is a read-modify-write, so the tallies need guarding or the
        # end-of-run "api calls / cache hits" line silently undercounts.
        self._stats_lock = threading.Lock()

    # ---------------------------------------------------------------- cache

    def _cache_path(self, image_bytes: bytes, prompt: str) -> Path | None:
        """Cache key covers image + prompt + model, so batched and single-pass
        runs never serve each other's answers."""
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(
            image_bytes + prompt.encode() + self.model.encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    # ---------------------------------------------------------------- call

    def _chat(self, content: list[dict]) -> str:
        """One chat call, with retry on transient failure and model fallback."""
        candidates = (
            [self.model]
            if self._model_confirmed
            else [self.model, *[m for m in MODEL_FALLBACKS if m != self.model]]
        )

        last_error: Exception | None = None
        for model in candidates:
            for attempt in range(self.max_retries):
                try:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stream=False,
                    )
                    self.model = model
                    self._model_confirmed = True
                    with self._stats_lock:
                        self.api_calls += 1
                    return resp.choices[0].message.content or ""
                except Exception as exc:  # noqa: BLE001 - SDK raises many types
                    last_error = exc
                    msg = str(exc).lower()
                    # Auth failure is terminal -- fail loudly, do not burn retries.
                    if any(
                        s in msg for s in ("unauthorized", "401", "forbidden", "403")
                    ):
                        raise CosmosError(f"Endpoint rejected the API key: {exc}") from exc
                    # An unknown model will never succeed on retry; try the next one.
                    if any(
                        s in msg
                        for s in ("not found", "does not exist", "unknown model", "404")
                    ):
                        break
                    time.sleep(min(2**attempt + random.random(), 20.0))

        raise CosmosError(
            f"All model candidates failed. Last error: {last_error}\n"
            f"Tried: {candidates}. Set POS_MODEL in .env to a model your key can reach."
        )

    # ---------------------------------------------------------------- api

    def detect(self, frame: Frame, frame_path: Path) -> list[Detection]:
        image_bytes = frame_path.read_bytes()
        data_uri = _b64_data_uri(frame_path)

        out: list[Detection] = []
        for prompt in self.prompts:
            cache = self._cache_path(image_bytes, prompt)
            if cache and cache.exists():
                try:
                    text = json.loads(cache.read_text())["text"]
                    with self._stats_lock:
                        self.cache_hits += 1
                    out.extend(parse_detections(text, frame, self.domain))
                    continue
                except (json.JSONDecodeError, KeyError):
                    pass  # corrupt cache entry, re-fetch

            text = self._chat(
                [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ]
            )
            if cache:
                cache.write_text(json.dumps({"text": text}))
            out.extend(parse_detections(text, frame, self.domain))

        # Batches cover disjoint class sets, so cross-batch duplicates are not
        # possible; a model repeating itself inside one reply is, so drop exact
        # class+box repeats.
        seen: set[tuple[str, int, int, int, int]] = set()
        unique: list[Detection] = []
        for d in out:
            key = (d.cls, *(int(round(v)) for v in d.box))  # type: ignore[misc]
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)
        return unique

    def probe(self) -> str:
        """Confirm the endpoint and model actually answer. Returns the model id."""
        self._chat([{"type": "text", "text": "Reply with the single word: ready"}])
        return self.model
