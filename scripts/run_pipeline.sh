#!/usr/bin/env bash
#
# The whole pipeline, start to end, on one clip.
#
#   bash scripts/run_pipeline.sh <video> <gpx> <camera> <run-name> [time-offset] [port]
#
# Example -- Kohima test_4, the run quoted in the README:
#   bash scripts/run_pipeline.sh \
#     road_videos/test_4/25545_NGRRDAAdmin_Kohima_*_170751_seg001.mp4 \
#     road_videos/test_4/25545_NGRRDAAdmin_Kohima_*_170751_seg001.gpx \
#     bike_kohima_t4 kohima4 -2.71 8090
#
# WHY A SCRIPT AND NOT JUST THE COMMANDS
# Two flags are easy to forget in ways that produce plausible-but-wrong output
# rather than an error: --time-offset (every finding displaced along the road)
# and --heading-baseline (findings thrown sideways on slow footage). Encoding
# them here means they cannot be skipped by accident.
#
# NOT the same as `pos run`, which is only step 2 of the six below.
#
# Get the time offset from step 1's output, then pass it as argument 5.

set -euo pipefail

VIDEO="${1:?usage: run_pipeline.sh <video> <gpx> <camera> <run-name> [time-offset] [port]}"
GPX="${2:?need a gpx}"
CAMERA="${3:?need a camera config name, e.g. bike_kohima_t4}"
NAME="${4:?need a run name}"
OFFSET="${5:-0}"
PORT="${6:-8090}"

OUT="runs/$NAME"

# The local defect model. Override with POS_ONNX=/path/to/model.onnx
ONNX="${POS_ONNX:-../../Cognecto/vision-stack/infrastructure/triton/models/post_cons/1/model.onnx}"

# The ensemble needs BOTH the ONNX model and an API key. If either is missing,
# degrade loudly -- a thinner result that looks like the full one is worse than
# a warning, because you cannot tell from the map that classes are missing.
BACKEND=ensemble
DOMAIN=road_pci

if [ ! -f "$ONNX" ]; then
  echo "!! ONNX model not found at:"
  echo "     $ONNX"
  echo "   Falling back to --backend cosmos (VLM only): ~4x fewer potholes, no PCI classes."
  echo "   Set POS_ONNX=/path/to/model.onnx for the full ensemble."
  BACKEND=cosmos
  DOMAIN=road
  ONNX=""
fi

if [ ! -f .env ] && [ -z "${NVIDIA_API_KEY:-}" ]; then
  echo "!! No .env file and no NVIDIA_API_KEY in the environment."
  if [ "$BACKEND" = ensemble ]; then
    echo "   Falling back to --backend onnx (local only): potholes yes, but no"
    echo "   waterlogging/garbage/hazards, and streetlight + footpath absence"
    echo "   cannot be inferred (the ONNX model has no such classes to miss)."
    BACKEND=onnx
  else
    echo "   Falling back to --backend mock: SYNTHETIC FIXTURES, not real perception."
    BACKEND=mock
  fi
fi

echo "=============================================================="
echo " video   : $VIDEO"
echo " gpx     : $GPX"
echo " camera  : $CAMERA"
echo " backend : $BACKEND      domain: $DOMAIN"
echo " offset  : $OFFSET s"
echo " out     : $OUT"
echo "=============================================================="

echo
echo "==> 1/6  PREFLIGHT  (pos doctor)"
# `|| true` on purpose: doctor exits non-zero on warnings as well as failures,
# and a time-offset warning is expected here -- we are about to pass the offset.
# Read its output; if it reports a DIFFERENT offset than you passed, re-run.
uv run pos doctor --video "$VIDEO" --gpx "$GPX" --camera "$CAMERA" || true

echo
echo "==> 2/6  PIPELINE  (ingest -> perceive -> localize -> cluster -> score -> twin)"
echo "    ONNX inference is ~2.3 s/frame on CPU, so budget a few minutes."
uv run pos run \
  --video "$VIDEO" --gpx "$GPX" \
  --camera "$CAMERA" --domain "$DOMAIN" --backend "$BACKEND" \
  ${ONNX:+--model-path "$ONNX"} \
  --fps 1 --time-offset "$OFFSET" --heading-baseline 15 \
  --out "$OUT"

echo
echo "==> 3/6  POINT CLOUD  (CPU monocular depth, ~1.5 s/frame)"
# Not fatal. The viewer falls back to the OSM twin, so a failure here costs
# a layer, not the run.
uv run pos depthcloud --run "$OUT" --camera "$CAMERA" --stride 5 \
  || echo "    point cloud failed or skipped -- the viewer will use the OSM twin"

echo
echo "==> 4/6  SATELLITE BASEMAP  (optional)"
# LICENCE: Esri World Imagery is free for NON-COMMERCIAL use and requires
# attribution, which the viewer renders from basemap.json. Set POS_TILE_PROVIDER
# or POS_TILE_URL for a different provider. Google's tiles cannot legally be used
# here -- that is what the KMZ export is for.
uv run pos basemap --run "$OUT" --provider "${POS_TILE_PROVIDER:-esri}" --zoom 18 \
  || echo "    basemap failed or skipped -- the viewer falls back to a plain plane"

echo
echo "==> 5/6  EXPORTS  (Google Earth + PDF)"
uv run pos kml --run "$OUT" --out "$NAME.kmz" || echo "    kmz export failed"
uv run pos report --run "$OUT" --out "$NAME.pdf" || echo "    pdf report failed"

echo
echo "==> 6/6  VIEWER"
if [ ! -d viewer/dist ]; then
  echo "    building viewer (first run only)"
  ( cd viewer && npm install && npm run build )
else
  echo "    viewer/dist already built"
fi

echo
echo "=============================================================="
uv run python - "$OUT" <<'PY'
import json, pathlib, sys
from collections import Counter

d = pathlib.Path(sys.argv[1])
m = json.loads((d / "manifest.json").read_text())
s = m.get("summary") or {}
print(f"  {m['n_frames']} keyframes | {s.get('route_length_m', 0):.0f} m | "
      f"{m['n_detections']} detections -> {m['n_findings']} findings")
print(f"  Index {s.get('quality_index')}/100   grade {s.get('grade')}")
print(f"  video: {m.get('has_video')}   point cloud: {m.get('has_pointcloud')}")
findings = json.loads((d / "findings.json").read_text())
for cls, n in Counter(f["cls"] for f in findings).most_common():
    print(f"    {n:>3}x {cls}")
PY
echo "=============================================================="
echo
echo " Serve it:"
echo "   uv run pos serve --run $OUT --port $PORT"
echo
echo " Then open  http://127.0.0.1:$PORT"
echo " Ctrl+Shift+R if you rebuilt the viewer -- the bundle name changes and"
echo " every run shares one origin cache."
echo
[ -f "$NAME.kmz" ] && echo " Google Earth:  $NAME.kmz   (Tools > Movie Maker records it to MP4)"
[ -f "$NAME.pdf" ] && echo " PDF report:    $NAME.pdf"
exit 0
