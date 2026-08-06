#!/usr/bin/env python
"""Checks the post_cons ONNX weights against the code that decodes them.

WHAT THIS IS GUARDING
Swapping models/post_cons/model.onnx is a two-line change that has three ways of
going silently wrong, and none of them raise:

  * WRONG COUNT     CLASS_KEYS is used both to slice the score rows
                    (pred[:, 4 : 4+len(CLASS_KEYS)]) and to look labels up by
                    index. Too few keys and the tail classes are never read; too
                    many and argmax indexes past the end of the head.
  * WRONG ORDER     Mapping is by index, so a retrain that reorders its classes
                    relabels every detection with no error. The 2-class weights
                    are exactly this trap: they put transverse at 0 and
                    longitudinal at 1, the reverse of the old 11-class model
                    (longitudinal 4, transverse 10).
  * WRONG TAXONOMY  A key the domain YAML does not define is dropped into
                    detector.skipped and never reported. A whole class can
                    vanish from a survey without one line of output.

Reads the model's own `names` metadata rather than trusting a comment, so this
fails the moment the weights and the code disagree. One session load, no network.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pos.config import DomainConfig  # noqa: E402
from pos.perception.onnx_yolo import CLASS_KEYS, resolve_model_path  # noqa: E402

FAILED: list[str] = []

#: Enough of each model class name to recognise it in our key, lowercased. The
#: exports are inconsistently cased ("Transverse crack" vs "longitudinal crack")
#: and our keys use underscores, so neither string matches the other outright.
STEM_LEN = 5


def chk(cond: bool, label: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILED.append(label)


def load_session():
    import onnxruntime as ort

    path = resolve_model_path(None)
    print(f"  model: {path}")
    # CPU only: this asserts on shapes and metadata, so a GPU provider would add
    # startup cost and a machine-dependent failure mode for no extra coverage.
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def check_head_width(session) -> None:
    """Score rows must be exactly 4 box rows plus one row per class key."""
    shape = session.get_outputs()[0].shape
    print(f"  output0 shape: {shape}")
    chk(len(shape) == 3, f"output is rank 3, got {len(shape)}")
    chk(
        shape[1] == 4 + len(CLASS_KEYS),
        f"head width {shape[1]} == 4 + {len(CLASS_KEYS)} class keys",
    )


def check_input_size(session) -> None:
    """The decoder letterboxes to a hardcoded INPUT_SIZE; the model must agree."""
    from pos.perception.onnx_yolo import INPUT_SIZE

    shape = session.get_inputs()[0].shape
    chk(
        list(shape[2:]) == [INPUT_SIZE, INPUT_SIZE],
        f"input {shape[2:]} matches INPUT_SIZE {INPUT_SIZE}",
    )


def check_class_order(session) -> None:
    """Our keys must name the same classes, in the same order, as the export."""
    raw = session.get_modelmeta().custom_metadata_map.get("names")
    if not raw:
        chk(False, "model carries a `names` metadata map")
        return

    # Ultralytics writes a Python dict repr, not JSON. literal_eval, not eval --
    # this is metadata from a file on disk, so it is an input, not a constant.
    names = ast.literal_eval(raw)
    print(f"  names: {names}")
    chk(
        len(names) == len(CLASS_KEYS),
        f"{len(names)} model classes == {len(CLASS_KEYS)} keys",
    )

    for i, key in enumerate(CLASS_KEYS):
        model_name = str(names.get(i, "")).lower()
        stem = model_name.split()[0][:STEM_LEN] if model_name else ""
        chk(
            bool(stem) and stem in key,
            f"index {i}: model {model_name!r} matches key {key!r}",
        )


def check_keys_in_domain() -> None:
    """Every key we can emit must exist in the domain, or detections are dropped."""
    dom = DomainConfig.load("road_pci")
    for key in CLASS_KEYS:
        chk(key in dom.class_map, f"road_pci.yaml defines {key!r}")


def check_severity_covers_keys() -> None:
    """severity_from falls back to 3 for an unknown key -- silently, so check."""
    import inspect

    from pos.perception.onnx_yolo import severity_from

    src = inspect.getsource(severity_from)
    for key in CLASS_KEYS:
        chk(f'"{key}"' in src, f"severity_from grades {key!r} explicitly")


def main() -> int:
    print("\npost_cons ONNX contract")
    print("=" * 64)
    session = load_session()
    check_head_width(session)
    check_input_size(session)
    check_class_order(session)
    check_keys_in_domain()
    check_severity_covers_keys()

    print("\n" + "=" * 64)
    if FAILED:
        print(f"  {len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("  Weights and decoder agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
