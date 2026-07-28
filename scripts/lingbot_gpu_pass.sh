#!/usr/bin/env bash
#
# One-off reconstruction pass: video -> real 3D point cloud, via lingbot-map.
# Written for a RunPod pod, but works on any CUDA box.
#
# WHY THIS IS A SEPARATE SCRIPT
# lingbot-map (Apache 2.0, github.com/Robbyant/lingbot-map) needs a CUDA GPU.
# PhysicalOS itself does not -- `pos depthcloud` already builds a usable cloud on
# CPU. This buys a better one: real multi-frame fusion with drift correction
# rather than stacked per-frame depth maps. Run it ONCE, keep the PLY.
#
# ---------------------------------------------------------------------------
# RUNPOD QUICKSTART
#
#   1. Pod: any 24 GB+ GPU (L4, A10, A5000, 4090, L40S, A100).
#      Template "RunPod PyTorch" is fine. 60 GB disk is plenty.
#
#   2. Get this script and your video onto the pod. runpodctl is preinstalled:
#
#        # ON YOUR LAPTOP -- prints a one-time code
#        runpodctl send road_videos/test_3/25545_*.mp4
#
#        # ON THE POD -- paste that code
#        runpodctl receive <CODE>
#
#      Or drag the file into the Jupyter file browser.
#
#   3. On the pod:
#        bash lingbot_gpu_pass.sh myvideo.mp4 out
#
#   4. Bring the predictions home. Only the NPZ matters, and the script tars
#      them for you:
#
#        # ON THE POD
#        runpodctl send out/predictions.tar.gz
#        # ON YOUR LAPTOP
#        runpodctl receive <CODE>
#
#   5. Back on your laptop:
#        mkdir -p lingbot_out && tar xzf predictions.tar.gz -C lingbot_out
#        uv run pos pointcloud --run runs/kohima_ens --preds lingbot_out
#        uv run pos serve --run runs/kohima_ens --port 8095
#      then tick the "Point cloud" layer.
#
# COST: 10-25 min of GPU for a 30 s clip, so well under $1 on most pods.
# ---------------------------------------------------------------------------

set -euo pipefail

VIDEO="${1:?usage: lingbot_gpu_pass.sh <video.mp4> [outdir]}"
OUT="${2:-lingbot_out}"
REPO="${LINGBOT_REPO:-$HOME/lingbot-map}"
CKPT_DIR="${LINGBOT_CKPT_DIR:-$HOME/lingbot-ckpt}"
CKPT="${LINGBOT_CKPT:-$CKPT_DIR/lingbot-map.pt}"

# Frames per second fed to the reconstruction. 10 gives good overlap; lower and
# the tracker loses the scene, higher just costs GPU time.
FPS="${FPS:-10}"

# SKIP_SMOKE=1 goes straight to the full run.
SKIP_SMOKE="${SKIP_SMOKE:-0}"

if [ ! -f "$VIDEO" ]; then
  echo "No such video: $VIDEO" >&2
  exit 1
fi

echo "=============================================================="
echo " lingbot-map reconstruction"
echo " video : $VIDEO"
echo " out   : $OUT"
echo "=============================================================="

echo
echo "==> 1/6  GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "No NVIDIA GPU visible. This script must run on a GPU box." >&2
  exit 1
}

echo
echo "==> 2/6  Repository"
if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 https://github.com/Robbyant/lingbot-map "$REPO"
else
  echo "    already cloned at $REPO"
fi

echo
echo "==> 3/6  Python environment"
# RunPod images ship python + pip with torch already installed, and usually no
# conda. So use conda only if it is genuinely present; otherwise install into the
# current interpreter. Pods are ephemeral, so isolation buys nothing.
if command -v conda >/dev/null 2>&1; then
  echo "    conda found -- using a dedicated env"
  conda env list | grep -q '^lingbot-map ' || conda create -y -n lingbot-map python=3.10
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lingbot-map
else
  echo "    no conda -- installing into the current interpreter ($(python3 -V 2>&1))"
fi

# lingbot-map's README pins torch 2.8.0 + cu128 because NVIDIA Kaolin only ships
# prebuilt wheels for that exact pair. Drifting off it means compiling Kaolin
# from source, which will eat your GPU hour. So check before touching anything.
HAVE_TORCH="$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
echo "    torch present: $HAVE_TORCH"
case "$HAVE_TORCH" in
  2.8.0*)
    echo "    correct version, leaving it alone"
    ;;
  none)
    echo "    installing torch 2.8.0+cu128 (large download)"
    pip install -q torch==2.8.0 torchvision==0.23.0 \
      --index-url https://download.pytorch.org/whl/cu128
    ;;
  *)
    echo "    WARNING: pod has torch $HAVE_TORCH, upstream pins 2.8.0+cu128."
    echo "    Trying as-is first -- reinstalling torch on a pod is slow and often"
    echo "    unnecessary. If the run fails on a Kaolin or CUDA error, re-run:"
    echo "      FORCE_TORCH=1 bash $0 $VIDEO $OUT"
    if [ "${FORCE_TORCH:-0}" = "1" ]; then
      echo "    FORCE_TORCH=1 -- installing the pinned pair"
      pip install -q torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128
    fi
    ;;
esac

echo "    installing lingbot-map"
pip install -q -e "$REPO"

# FlashInfer powers the paged-KV-cache attention. If it will not install we fall
# back to SDPA: slower, but correct.
USE_SDPA=""
if pip install -q flashinfer-python 2>/dev/null; then
  echo "    flashinfer ok"
else
  echo "    flashinfer unavailable -- will pass --use_sdpa"
  USE_SDPA="--use_sdpa"
fi
pip install -q onnxruntime-gpu huggingface_hub || true

echo
echo "==> 4/6  Checkpoint"
mkdir -p "$CKPT_DIR"
# TWO files are needed, not one: outdoor_drive.yaml sets
# `sky_model: skyseg_batch.onnx`, so sky masking fails without it.
if [ ! -f "$CKPT" ] || [ ! -f "$CKPT_DIR/skyseg_batch.onnx" ]; then
  echo "    downloading lingbot-map.pt + skyseg_batch.onnx (not gated, no token)"
  python3 - "$CKPT_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
dest = sys.argv[1]
for fn in ("lingbot-map.pt", "skyseg_batch.onnx"):
    print("   ", hf_hub_download("robbyant/lingbot-map", fn, local_dir=dest))
PY
else
  echo "    already present"
fi
ls -lh "$CKPT_DIR" | awk 'NR>1 {printf "    %-28s %s\n", $9, $5}'

CFG="$REPO/demo_render/config/outdoor_drive.yaml"
[ -f "$CFG" ] || { echo "Missing $CFG -- did the clone succeed?" >&2; exit 1; }

# outdoor_drive.yaml is the upstream preset for road footage: max_depth 250 m,
# sky masking on, follow-then-birdeye camera, 1080p render. The indoor defaults
# clamp depth far too close for a street.
COMMON=(
  --video_path "$VIDEO"
  --model_path "$CKPT"
  --config "$CFG"
  --skyseg_model_path "$CKPT_DIR/skyseg_batch.onnx"
  --fps "$FPS"
  --mode windowed
  --window_size 128
  --overlap_keyframes 16
  --keyframe_interval 10
  --mask_sky
  --keyframes_only_points
  --save_predictions
)

if [ "$SKIP_SMOKE" != "1" ]; then
  echo
  echo "==> 5/6  Smoke test on the first 20 frames"
  echo "    Proving the environment works before committing the whole clip: a"
  echo "    failure here costs seconds, the same failure 20 minutes in does not."
  if python3 "$REPO/demo_render/batch_demo.py" \
      "${COMMON[@]}" $USE_SDPA \
      --output_folder "${OUT}_smoke" \
      --first_k 20; then
    n=$(find "${OUT}_smoke" -name '*.npz' | wc -l)
    echo "    smoke test OK -- $n npz files"
    [ "$n" -gt 0 ] || {
      echo "    but no NPZ was written; --save_predictions did nothing" >&2
      exit 1
    }
  else
    echo
    echo "SMOKE TEST FAILED. Nothing wasted. Common causes:" >&2
    echo "  - torch/CUDA mismatch  -> FORCE_TORCH=1 bash $0 $VIDEO $OUT" >&2
    echo "  - Kaolin build error   -> same fix; the pinned pair has wheels" >&2
    echo "  - out of GPU memory    -> bigger pod, or --window_size 64" >&2
    exit 1
  fi
else
  echo
  echo "==> 5/6  Smoke test skipped (SKIP_SMOKE=1)"
fi

echo
echo "==> 6/6  Full reconstruction (the slow part)"
mkdir -p "$OUT"
time python3 "$REPO/demo_render/batch_demo.py" \
  "${COMMON[@]}" $USE_SDPA \
  --output_folder "$OUT"

NPZ=$(find "$OUT" -name '*.npz' | wc -l)
echo
echo "=============================================================="
echo " Done. $NPZ npz prediction files under $OUT"
find "$OUT" -name '*.npz' | head -3 | sed 's/^/   /'
du -sh "$OUT" | awk '{print "   total size: "$1}'

# Tar only the NPZ: the rendered MP4s are large and PhysicalOS does not need
# them, so this keeps the transfer home small.
echo
echo " Packaging predictions for transfer ..."
( cd "$OUT" && find . -name '*.npz' -print0 | tar czf predictions.tar.gz --null -T - ) \
  || tar czf "$OUT/predictions.tar.gz" -C "$OUT" .
du -sh "$OUT/predictions.tar.gz" | awk '{print "   "$1"  "$2}'

cat <<EOF

--------------------------------------------------------------
NEXT: get it home and build the cloud

  ON THE POD
    runpodctl send $OUT/predictions.tar.gz

  ON YOUR LAPTOP
    runpodctl receive <CODE>
    mkdir -p lingbot_out && tar xzf predictions.tar.gz -C lingbot_out

    uv run pos pointcloud --run runs/kohima_ens --preds lingbot_out
    uv run pos serve --run runs/kohima_ens --port 8095

  Then tick the "Point cloud" layer in the viewer.

  pos pointcloud fits lingbot's camera trajectory to your GPX with a
  similarity transform, so the cloud lands in the same local ENU metres
  as the findings and the OSM twin.

LICENCE: lingbot-map is Apache-2.0 -- fine commercially, unlike the AGPL
YOLO weights. Attribute Robbyant/lingbot-map.
--------------------------------------------------------------
EOF
