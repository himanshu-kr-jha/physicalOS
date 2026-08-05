#!/usr/bin/env bash
#
# One-shot install. Run once on a fresh clone, then use ./start.sh.
#
#   bash setup.sh
#
# This is the README quickstart made runnable, plus the two things prose cannot do:
# check that ffmpeg is really on PATH, and report which perception backends this
# machine can actually use. Both are otherwise discovered several minutes into a
# failing upload.

set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m+\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- prerequisites
# Checked before anything is installed, so a missing system package is one clear
# message rather than a traceback out of ffmpeg-not-found halfway through ingest.
say "1/5  Checking prerequisites"

MISSING=0
need() {
  if command -v "$1" >/dev/null 2>&1; then
    ok "$1"
  else
    warn "$1 NOT FOUND -- $2"
    MISSING=1
  fi
}

need uv      "https://docs.astral.sh/uv/  (curl -LsSf https://astral.sh/uv/install.sh | sh)"
need node    "Node 20+ from https://nodejs.org or your package manager"
need npm     "ships with Node"
# ffmpeg AND ffprobe: `pos ingest` extracts keyframes with one and reads the clip's
# duration and creation_time with the other. Minimal builds sometimes omit ffprobe,
# and the failure that causes looks like a corrupt video rather than a missing tool.
need ffmpeg  "apt install ffmpeg   (keyframe extraction)"
need ffprobe "apt install ffmpeg   (video duration + creation_time)"

if [ "$MISSING" -ne 0 ]; then
  printf '\n\033[31mMissing prerequisites above. Install them, then re-run.\033[0m\n\n'
  exit 1
fi

# ------------------------------------------------------------------ python deps
say "2/5  Python environment  (uv sync)"
uv sync
ok "python deps installed into .venv"

# ------------------------------------------------------------------ viewer deps
# The viewer is a real build, not a static file: FastAPI mounts viewer/dist and
# serves nothing at / until this has run at least once.
say "3/5  Viewer bundle  (npm install && npm run build)"
( cd viewer && npm install && npm run build )
ok "viewer/dist built"

# ------------------------------------------------------------------------- .env
say "4/5  Configuration"
if [ -f .env ]; then
  ok ".env already present -- left untouched"
else
  cp .env.example .env
  ok ".env created from .env.example"
  warn "it holds a PLACEHOLDER key. Edit it to use the VLM backends (cosmos/ensemble)."
fi

# `pos serve --studio` writes runs here and stages uploads here. Creating them now
# keeps the first upload from racing the directory into existence.
mkdir -p runs uploads
ok "runs/ and uploads/ ready"

# ------------------------------------------------------------------- capability
# State what this machine can actually do, using the SAME resolver the upload form
# calls, so the two can never disagree about what is available.
say "5/5  What this machine can run"
uv run python - <<'PY'
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(".env"))
except ImportError:
    pass

def ok(m):   print(f"  \033[32m+\033[0m {m}")
def warn(m): print(f"  \033[33m!\033[0m {m}")

key = os.environ.get("NVIDIA_API_KEY", "")
if key.startswith("nvapi-") and "xxxx" not in key:
    ok("NVIDIA_API_KEY set         -> backends: cosmos, ensemble")
else:
    warn("no usable NVIDIA_API_KEY   -> backends cosmos/ensemble unavailable")

# resolve_model_path is what /api/cameras uses to answer has_onnx, so this line and
# the studio form always tell the same story about the local detector.
try:
    from pos.perception.onnx_yolo import resolve_model_path
    p = resolve_model_path(None)
    if p:
        ok(f"local ONNX detector        -> {p}")
    else:
        warn("no ONNX model found        -> onnx/ensemble unavailable (set POS_ONNX)")
except Exception as exc:  # noqa: BLE001 - a report must never fail the install
    warn(f"ONNX detector unavailable  ({type(exc).__name__}: {exc})")

try:
    from pos.segment import resolve_model_path as road_model
    ok(f"road segmentation model    -> {road_model(None)}")
except Exception:  # noqa: BLE001
    warn("no road segmentation model -> the carriageway mask layer is skipped")

print()
print("  `mock` always works and needs nothing -- synthetic fixtures, useful for")
print("  exercising the UI but NOT real perception.")
print("  A findings CSV needs no backend at all: attach one in the studio and its")
print("  rows become the findings directly.")
PY

say "Done."
cat <<'EOF'
  Start it:    bash start.sh
  Then open:   http://127.0.0.1:8090/studio    upload a video + GPX (+ optional CSV)
               http://127.0.0.1:8090/          3D viewer

  Verify the install against the committed ground truth:
    uv run python scripts/make_sample.py && uv run python scripts/verify_sample.py
    uv run python scripts/test_import_csv.py

EOF
