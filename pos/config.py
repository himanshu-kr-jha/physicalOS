"""Loads camera calibration and domain taxonomies from YAML.

The taxonomy is DATA, not code. Adding a new inspection domain (construction
safety, facade survey, rail, pipeline...) is a new YAML file -- no Python
changes. This is what lets one engine serve every infrastructure type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass
class CameraConfig:
    """Monocular calibration. Drives 2D-box -> ground-plane projection."""

    height_m: float = 1.4
    vfov_deg: float = 58.0
    hfov_deg: float = 90.0
    # Positive pushes the horizon DOWN the image (camera pitched up).
    # Fraction of image height, relative to centre.
    pitch_offset_frac: float = 0.0
    max_range_m: float = 60.0

    @classmethod
    def load(cls, name_or_path: str = "dashcam") -> CameraConfig:
        path = Path(name_or_path)
        if not path.exists():
            path = CONFIG_DIR / "camera" / f"{name_or_path}.yaml"
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ClassSpec:
    """One thing we look for in the world."""

    key: str
    label: str
    geometry: str = "point"          # point | segment | area
    cluster_radius_m: float = 3.0
    weight: float = 1.0              # contribution to the quality penalty
    alert: bool = False              # surface as a real-time alert
    hint: str = ""                   # prompt guidance for the VLM
    color: str = "#f59e0b"


@dataclass
class AbsenceSpec:
    """A defect defined by something NOT being there.

    Absence cannot be detected in a single frame -- asking a model "is there a
    lighting gap?" asks it to reason about what it cannot see, and it guesses.
    It has to be INFERRED from coverage: if no streetlight was detected anywhere
    along 60 m of route, that silence is the evidence.

    So each rule names a presence class (`asset`) and a distance (`min_gap_m`).
    pos/absence.py walks the route and emits a finding for every stretch longer
    than that which contains none of the asset.
    """

    key: str                 # the absence class emitted, e.g. streetlight_missing
    label: str
    asset: str               # the presence class whose absence we infer
    min_gap_m: float = 30.0
    weight: float = 1.5
    severity: int = 3
    color: str = "#78716c"
    alert: bool = False
    note: str = ""           # shown in the evidence panel


@dataclass
class DomainConfig:
    key: str
    label: str
    description: str = ""
    prompt_context: str = ""
    index_name: str = "Quality Index"
    classes: list[ClassSpec] = field(default_factory=list)
    absences: list[AbsenceSpec] = field(default_factory=list)

    @property
    def class_map(self) -> dict[str, ClassSpec]:
        """Every scorable class, including the synthesised absence ones.

        Absence keys are turned into real ClassSpecs so clustering, scoring and
        the viewer legend treat them like any other class -- no special cases
        scattered through the pipeline.
        """
        out = {c.key: c for c in self.classes}
        for a in self.absences:
            out.setdefault(
                a.key,
                ClassSpec(
                    key=a.key,
                    label=a.label,
                    geometry="segment",
                    cluster_radius_m=max(a.min_gap_m, 15.0),
                    weight=a.weight,
                    alert=a.alert,
                    color=a.color,
                    hint=a.note,
                ),
            )
        return out

    def spec(self, key: str) -> ClassSpec:
        return self.class_map.get(
            key, ClassSpec(key=key, label=key.replace("_", " ").title())
        )

    @classmethod
    def load(cls, name_or_path: str = "road") -> DomainConfig:
        path = Path(name_or_path)
        if not path.exists():
            path = CONFIG_DIR / "domains" / f"{name_or_path}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"No domain config for {name_or_path!r}. Available: {available_domains()}"
            )
        data = yaml.safe_load(path.read_text()) or {}
        classes = [ClassSpec(**c) for c in data.pop("classes", [])]
        absences = [AbsenceSpec(**a) for a in data.pop("absence", [])]
        known = {"key", "label", "description", "prompt_context", "index_name"}
        return cls(
            **{k: v for k, v in data.items() if k in known},
            classes=classes,
            absences=absences,
        )


def available_domains() -> list[str]:
    d = CONFIG_DIR / "domains"
    return sorted(p.stem for p in d.glob("*.yaml")) if d.exists() else []
