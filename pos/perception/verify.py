"""Second-pass confirmation: re-ask "is this really X?" about one crop.

THE PROBLEM IT ADDRESSES
Measured on the synthetic sample, single-pass perception runs at 64% precision and
48% recall. Both numbers have one cause: a model handed a whole frame and a long
class list answers in a single open-ended shot, and open-ended shots are where
false positives come from. Two existing measurements point the way:

  - filtering to findings with >= 2 sightings lifted precision 64% -> 83%. Repeat
    observation is therefore already a strong precision signal, which means the
    findings worth scrutinising are the SINGLETONS.
  - `--classes-per-call 4` lifted recall 48% -> 55% but halved precision, because
    asking "is there a hazard?" biases toward yes (12 hazards reported where 1
    existed). It is off by default for that reason. A verification pass is what
    would make it usable: recover recall with batched prompts, then spend a cheap
    second call removing the junk.

WHY THIS OPERATES ON FINDINGS, NOT RAW DETECTIONS
The plan said detections, between `perceive` and `localize`. Findings turned out
to be the better unit. "Singleton" is only meaningful after clustering -- a lone
detection cannot know whether the same object was seen again two frames later.
And a finding already knows its clearest sighting, which is the image worth asking
about; the first sighting is usually a few-pixel sliver 30 m down the road. So
this runs after `cluster` and before `score`.

AVOIDING THE POLITENESS TRAP
"Is this a pothole?" invites yes. The prompt therefore names the specific
lookalikes to reject, offers an explicit "unsure" so a blurry crop need not become
a yes or a no, and states plainly that a wrong confirmation is worse than a
rejection. Absence findings are never verified -- there is nothing in the frame to
look at, which is the entire point of them.

COST
Only the selected subset is checked, so a 57-finding run costs roughly 20-30 calls
rather than 57. A cropped image plus a different prompt hashes to a different
cache key (CosmosDetector._cache_path covers image+prompt+model), so verification
caches independently of detection and a re-run is free.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from ..config import DomainConfig
from ..schema import BOX_SCALE, Detection, Finding
from .base import _extract_json

# Context to include around the box. A bare crop of a pothole is hard to read;
# the surrounding tarmac is what makes it recognisable as a cavity, not a stain.
PAD_FRAC = 0.30

# Crops narrower than this get upscaled. The model resizes its input anyway, and
# a 40x30 crop sent at 40x30 is illegible.
MIN_CROP_PX = 320

_CONTRACT = """
Answer with ONLY this JSON, no prose:
{"verdict": "confirm" | "reject" | "unsure",
 "confidence": <0.0-1.0>,
 "reason": "<one short sentence describing what you actually see>"}
"""


@dataclass
class Verdict:
    finding_id: str
    cls: str
    verdict: str  # confirm | reject | unsure
    confidence: float
    reason: str


def _crop_data_uri(frame_path: Path, det: Detection) -> str | None:
    """The detection's box plus context padding, as a base64 JPEG data URI."""
    if not frame_path.exists():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(frame_path) as im:
            img = im.convert("RGB")
            w, h = img.size
            x1 = det.box[0] / BOX_SCALE * w
            y1 = det.box[1] / BOX_SCALE * h
            x2 = det.box[2] / BOX_SCALE * w
            y2 = det.box[3] / BOX_SCALE * h

            pw, ph = (x2 - x1) * PAD_FRAC, (y2 - y1) * PAD_FRAC
            box = (
                int(max(0, x1 - pw)),
                int(max(0, y1 - ph)),
                int(min(w, x2 + pw)),
                int(min(h, y2 + ph)),
            )
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                return None

            crop = img.crop(box)
            if crop.width < MIN_CROP_PX:
                scale = MIN_CROP_PX / crop.width
                crop = crop.resize(
                    (MIN_CROP_PX, max(1, int(crop.height * scale))), Image.LANCZOS
                )

            buf = BytesIO()
            crop.save(buf, "JPEG", quality=88)
            return (
                "data:image/jpeg;base64,"
                + base64.b64encode(buf.getvalue()).decode()
            )
    except OSError:
        return None


def build_verify_prompt(label: str, hint: str = "") -> str:
    """A deliberately sceptical single-question prompt."""
    hint = " ".join(hint.split()) if hint else ""
    return "\n".join(
        [
            "You are auditing one region of a road-survey photograph. The image "
            "below is a CROP of that region, with some surrounding context.",
            "",
            f"A previous detector claims this region shows: {label}."
            + (f" It should look like: {hint}" if hint else ""),
            "",
            "Decide whether that claim is correct.",
            "",
            "- confirm  only if you can actually SEE the described thing here.",
            "- reject   if the region instead shows a shadow, a dark stain, a wet "
            "patch, a repaired patch, a manhole or utility cover, a road marking, "
            "vegetation, or simply intact undamaged road.",
            "- unsure   if the crop is too blurry, dark, small or ambiguous to "
            "judge. This is a valid answer -- use it rather than guessing.",
            "",
            "Do not confirm out of politeness or to be agreeable. A wrong "
            "confirmation puts a false defect on an engineer's map, which is "
            "worse than rejecting a real one.",
            _CONTRACT,
        ]
    )


def needs_check(
    f: Finding,
    absence_keys: set[str],
    max_conf: float = 0.85,
) -> bool:
    """Should this finding be re-examined?

    Selective on purpose: verifying everything would roughly double a run's API
    cost for no gain on findings that are already well evidenced.
    """
    if f.cls in absence_keys:
        return False  # inferred from coverage; no pixels to inspect
    if not f.evidence:
        return False
    if len(f.evidence) >= 2:
        return False  # repeat observation already implies ~83% precision
    return f.confidence < max_conf


def verify_findings(
    findings: list[Finding],
    run_dir: Path,
    detector,
    domain: DomainConfig,
    unsure_penalty: float = 0.75,
    confirm_boost: float = 1.15,
    progress: bool = True,
) -> tuple[list[Finding], list[Verdict], dict]:
    """Confirm or drop the shakier findings. Returns (kept, verdicts, stats).

    `detector` must be a CosmosDetector: its _chat and _cache_path carry the
    retry, model-fallback and caching behaviour, and a second implementation here
    would be a second thing to keep correct.
    """
    absence_keys = {a.key for a in domain.absences}
    class_map = domain.class_map

    kept: list[Finding] = []
    verdicts: list[Verdict] = []
    skipped = 0
    todo = sum(1 for f in findings if needs_check(f, absence_keys))
    done = 0

    for f in findings:
        if not needs_check(f, absence_keys):
            kept.append(f)
            continue

        det = f.evidence[0]
        uri = _crop_data_uri(run_dir / "frames" / f"{det.frame_id}.jpg", det)
        if uri is None:
            skipped += 1
            kept.append(f)  # cannot check it, so do not silently delete it
            continue

        spec = class_map.get(f.cls)
        prompt = build_verify_prompt(
            f.label or (spec.label if spec else f.cls),
            spec.hint if spec else "",
        )

        crop_bytes = base64.b64decode(uri.split(",", 1)[1])
        cache = detector._cache_path(crop_bytes, prompt)
        text = None
        if cache and cache.exists():
            try:
                text = json.loads(cache.read_text())["text"]
                detector.cache_hits += 1
            except (json.JSONDecodeError, KeyError):
                text = None
        if text is None:
            text = detector._chat(
                [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": prompt},
                ]
            )
            if cache:
                cache.write_text(json.dumps({"text": text}))

        obj = _extract_json(text) or {}
        raw = str(obj.get("verdict", "")).strip().lower()
        # Anything unrecognised is treated as "unsure", never as a confirmation.
        verdict = raw if raw in ("confirm", "reject", "unsure") else "unsure"
        try:
            conf = float(obj.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        reason = str(obj.get("reason", "")).strip()[:300]

        verdicts.append(Verdict(f.finding_id, f.cls, verdict, conf, reason))
        done += 1
        if progress and (done % 10 == 0 or done == todo):
            print(f"  verified {done}/{todo}")

        if verdict == "reject":
            continue  # dropped from the register

        adjusted = f.model_copy(deep=True)
        if verdict == "confirm":
            adjusted.confidence = min(1.0, round(f.confidence * confirm_boost, 3))
            tag = f"[verified: {reason}]" if reason else "[verified]"
        else:
            adjusted.confidence = max(0.05, round(f.confidence * unsure_penalty, 3))
            tag = f"[unconfirmed: {reason}]" if reason else "[unconfirmed]"
        # Keep both models' words. A human comparing them is exactly the audit
        # trail this project exists to provide.
        adjusted.evidence[0].evidence = f"{det.evidence} {tag}".strip()
        kept.append(adjusted)

    stats = {
        "input": len(findings),
        "checked": len(verdicts),
        "confirmed": sum(1 for v in verdicts if v.verdict == "confirm"),
        "rejected": sum(1 for v in verdicts if v.verdict == "reject"),
        "unsure": sum(1 for v in verdicts if v.verdict == "unsure"),
        "skipped_no_image": skipped,
        "kept": len(kept),
    }
    return kept, verdicts, stats
