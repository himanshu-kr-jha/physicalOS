"""Optional box refiner using nvidia/LocateAnything-3B.

*** LICENCE WARNING ***
LocateAnything-3B is released under the "NVIDIA License for non-commercial
use": academic and non-profit research purposes only. Commercial use is
prohibited. This backend is therefore:

  - never the default,
  - gated behind POS_ACCEPT_NONCOMMERCIAL=1,
  - not for anything customer-facing or investor-facing.

Cosmos Reason (pos/perception/cosmos.py) carries the runtime work precisely so
that this file stays optional.

*** HARDWARE ***
Needs an NVIDIA Ampere/Hopper/Lovelace/Blackwell GPU and roughly 12 GB of VRAM
with optimised backends, up to ~35 GB with plain SDPA. It cannot run on a
machine without a CUDA GPU, which is why it is imported lazily -- importing
this module is what pulls in torch.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ..config import DomainConfig
from ..schema import BOX_SCALE, Detection, Frame

MODEL_ID = "nvidia/LocateAnything-3B"

# The model emits <box><x1><y1><x2><y2></box> with values already on a 0-1000
# scale, which is the same convention as pos.schema.BOX_SCALE.
_BOX_RE = re.compile(
    r"<box>\s*<?(-?\d+(?:\.\d+)?)>?\s*<?(-?\d+(?:\.\d+)?)>?\s*"
    r"<?(-?\d+(?:\.\d+)?)>?\s*<?(-?\d+(?:\.\d+)?)>?\s*</box>"
)


class LocateAnythingError(RuntimeError):
    pass


class LocateAnythingDetector:
    """Re-localises classes that another detector has already named.

    This is a REFINER, not a discoverer. It is good at "where exactly is the
    thing you described", not at deciding what is worth reporting -- so it
    grounds one phrase at a time rather than being asked to survey a scene.
    """

    name = "locate-anything"

    def __init__(self, domain: DomainConfig, max_new_tokens: int = 512):
        if os.environ.get("POS_ACCEPT_NONCOMMERCIAL") != "1":
            raise LocateAnythingError(
                f"{MODEL_ID} is licensed for NON-COMMERCIAL research use only.\n"
                "If that applies to you, set POS_ACCEPT_NONCOMMERCIAL=1 to proceed.\n"
                "For any commercial demo use `--backend cosmos` instead."
            )

        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise LocateAnythingError(
                "This backend needs torch and transformers:\n"
                "  uv pip install torch transformers accelerate"
            ) from exc

        if not torch.cuda.is_available():
            raise LocateAnythingError(
                "No CUDA GPU detected. LocateAnything-3B needs an NVIDIA GPU with "
                "~12 GB+ of VRAM. Use `--backend cosmos` (hosted, no GPU) instead."
            )

        self.torch = torch
        self.domain = domain
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .to("cuda")
            .eval()
        )

    # ------------------------------------------------------------------

    def _ground(self, image, phrase: str) -> list[list[float]]:
        """Ask for every instance of one phrase. Returns boxes on the 0-1000 scale."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Locate every {phrase}."},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to("cuda")

        with self.torch.no_grad():
            out = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.torch.bfloat16),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                generation_mode="hybrid",
            )

        reply = (
            out
            if isinstance(out, str)
            else self.tokenizer.decode(out[0], skip_special_tokens=True)
        )

        boxes: list[list[float]] = []
        for m in _BOX_RE.finditer(reply):
            vals = [float(v) for v in m.groups()]
            x1, x2 = sorted((vals[0], vals[2]))
            y1, y2 = sorted((vals[1], vals[3]))
            box = [
                max(0.0, min(BOX_SCALE, x1)),
                max(0.0, min(BOX_SCALE, y1)),
                max(0.0, min(BOX_SCALE, x2)),
                max(0.0, min(BOX_SCALE, y2)),
            ]
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                boxes.append(box)
        return boxes

    def detect(self, frame: Frame, frame_path: Path) -> list[Detection]:
        from PIL import Image

        image = Image.open(frame_path).convert("RGB")

        out: list[Detection] = []
        for spec in self.domain.classes:
            phrase = spec.label.lower()
            for box in self._ground(image, phrase):
                out.append(
                    Detection(
                        frame_id=frame.frame_id,
                        cls=spec.key,
                        box=box,
                        severity=3,  # this model localises; it does not grade
                        confidence=0.7,
                        evidence=(
                            f"Grounded by LocateAnything-3B from the phrase "
                            f"'{phrase}'. Severity not assessed by this model."
                        ),
                    )
                )
        return out
