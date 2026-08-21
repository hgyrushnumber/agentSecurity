#!/usr/bin/env bash
# Download a model from HuggingFace into models/.
#
# Usage:
#   bash scripts/download_model.sh                      # default Qwen/Qwen3-4B
#   bash scripts/download_model.sh Qwen/Qwen2.5-1.5B-Instruct
#   MODEL_DIR=/data/models bash scripts/download_model.sh Qwen/Qwen3-4B
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

MODEL_NAME=Qwen/Qwen3-4B
if [ "$#" -ge 1 ]; then
  MODEL_NAME=$1
fi
MODEL_DIR=${MODEL_DIR:-./models/$(basename "$MODEL_NAME")}

echo "================================="
echo "Download Model: $MODEL_NAME"
echo "Target dir:     $MODEL_DIR"
echo "================================="

if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  echo "Model already exists: $MODEL_DIR (skip)"
  exit 0
fi

mkdir -p models
python - <<PYEOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$MODEL_NAME",
    local_dir="$MODEL_DIR",
)
print("downloaded to", "$MODEL_DIR")
PYEOF
