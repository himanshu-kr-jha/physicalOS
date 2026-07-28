"""The detector interface, prompt construction, and strict-JSON parsing.

All backends -- hosted Cosmos Reason, the offline mock, an optional local
LocateAnything refiner -- implement the same `Detector` protocol and must
return boxes in the one canonical convention (0..1000, origin top-left).

Keeping the parsing here means a flaky model reply degrades to "no detections
in this frame" rather than crashing a 40-minute run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import DomainConfig
from ..schema import BOX_SCALE, Detection, Frame


@runtime_checkable
class Detector(Protocol):
    """Anything that can look at a frame and report what it sees."""

    name: str

    def detect(self, frame: Frame, frame_path: Path) -> list[Detection]:
        ...


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

_JSON_CONTRACT = """
Return ONLY a JSON object, no prose, no markdown fence:
{"detections": [{"cls": "<class key>", "box": [x1, y1, x2, y2],
                 "severity": <1-5>, "confidence": <0.0-1.0>,
                 "evidence": "<one short sentence of visual justification>"}]}

Rules:
- box coordinates are integers 0-1000, normalised to image width and height,
  origin at the TOP-LEFT corner. x rightward, y downward.
- cls MUST be one of the class keys listed above. Never invent a class.
- severity: 1 = cosmetic, 3 = needs attention, 5 = urgent / unsafe.
- confidence: how sure you are the object is really there and really that class.
- evidence: state what you actually SEE that justifies the call. This is shown
  to a human inspector next to the image, so be concrete and specific.
- If you see nothing from the list, return {"detections": []}. Do not guess.
"""


def build_prompt(domain: DomainConfig, subset: list[str] | None = None) -> str:
    """Assemble the instruction from the domain YAML. No hardcoded taxonomy.

    `subset` restricts the prompt to specific class keys. Asking about a few
    classes at a time measurably improves recall: handed the full list in one
    open-ended pass, models volunteer only the famous defect types (potholes,
    cracks, streetlights) and silently skip the rest. Naming a short list forces
    each one to be considered -- in testing this recovered hazards and footpath
    obstructions that the single-pass prompt never mentioned at all.

    See CosmosDetector(classes_per_call=...).
    """
    classes = domain.classes
    if subset:
        wanted = set(subset)
        classes = [c for c in classes if c.key in wanted]

    lines = [domain.prompt_context.strip(), ""]
    if subset:
        labels = ", ".join(c.label for c in classes)
        lines.append(
            f"Check this image specifically for: {labels}. "
            "Consider each one in turn and report every instance you can see."
        )
        lines.append("")
    lines.append("Classes to report:")
    for c in classes:
        hint = " ".join(c.hint.split()) if c.hint else c.label
        lines.append(f"- {c.key} ({c.label}): {hint}")
    lines.append(_JSON_CONTRACT)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a model reply that may be wrapped in prose.

    Reasoning models like to narrate before answering, and often wrap the
    payload in a markdown fence. Try progressively looser strategies.
    """
    text = text.strip()
    if not text:
        return None

    # Reasoning models may emit a <think>...</think> preamble.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()

    for candidate in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        try:
            obj = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"detections": obj}

    # Last resort: the outermost brace pair.
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _coerce_box(raw: object) -> list[float] | None:
    """Validate and normalise a box into [x1,y1,x2,y2] within 0..BOX_SCALE."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if any(v != v for v in vals):  # NaN
        return None

    # Some models answer in 0..1 despite instructions. Detect and rescale.
    if all(0.0 <= v <= 1.0 for v in vals):
        vals = [v * BOX_SCALE for v in vals]

    x1, y1, x2, y2 = vals
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    box = [
        max(0.0, min(BOX_SCALE, x1)),
        max(0.0, min(BOX_SCALE, y1)),
        max(0.0, min(BOX_SCALE, x2)),
        max(0.0, min(BOX_SCALE, y2)),
    ]
    # Reject degenerate slivers -- usually a hallucinated coordinate.
    if box[2] - box[0] < 2.0 or box[3] - box[1] < 2.0:
        return None

    # Reject whole-frame boxes. On real photographs models often recognise the
    # defect correctly and then box the entire image -- observed on 2 of 5 real
    # pothole photos, returning [0,0,999,999]. The class is right and the box is
    # worthless: its bottom edge is the image bottom, so ground-plane projection
    # would place the finding a couple of metres ahead of the camera regardless
    # of the truth. Dropping it beats inventing a position.
    #
    # BOTH axes must be near-full to trigger, so a genuinely road-wide but
    # shallow crack, or a tall narrow pole, still passes.
    if (box[2] - box[0]) > 0.92 * BOX_SCALE and (box[3] - box[1]) > 0.92 * BOX_SCALE:
        return None

    return box


def parse_detections(text: str, frame: Frame, domain: DomainConfig) -> list[Detection]:
    """Turn a raw model reply into validated Detections. Never raises."""
    obj = _extract_json(text)
    if not obj:
        return []

    items = obj.get("detections")
    if not isinstance(items, list):
        return []

    valid_keys = set(domain.class_map)
    out: list[Detection] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cls = str(
            item.get("cls") or item.get("class") or item.get("label") or ""
        ).strip()
        if cls not in valid_keys:
            continue  # silently drop invented classes
        box = _coerce_box(item.get("box") or item.get("bbox") or item.get("box_2d"))
        if box is None:
            continue
        try:
            severity = int(round(float(item.get("severity", 3))))
        except (TypeError, ValueError):
            severity = 3
        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6

        out.append(
            Detection(
                frame_id=frame.frame_id,
                cls=cls,
                box=box,
                severity=max(1, min(5, severity)),
                confidence=max(0.0, min(1.0, confidence)),
                evidence=str(
                    item.get("evidence") or item.get("reason") or ""
                ).strip()[:400],
            )
        )
    return out
