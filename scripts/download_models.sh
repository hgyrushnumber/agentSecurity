#!/usr/bin/env bash
# Download one or more models from configs/models.json.
#
# Usage:
#   bash scripts/download_models.sh list
#   bash scripts/download_models.sh qwen3_4b mistral_7b
#   bash scripts/download_models.sh all
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/download_models.sh list|all|MODEL_ID..." >&2
  exit 1
fi

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  echo "warning: 'hf' is unavailable; falling back to deprecated 'huggingface-cli'" >&2
  echo "install or upgrade huggingface_hub to use the supported CLI" >&2
  HF_DOWNLOAD=(huggingface-cli download)
else
  echo "Missing Hugging Face CLI. Install it with:" >&2
  echo "  python -m pip install --upgrade huggingface_hub" >&2
  exit 127
fi

if [ "$1" = "list" ]; then
  python -m sft.model_registry list
  exit 0
fi

if [ "$1" = "all" ]; then
  MODEL_IDS=()
  while read -r MODEL_ID; do
    MODEL_IDS+=("$MODEL_ID")
  done < <(python -m sft.model_registry list | awk '{print $1}')
else
  MODEL_IDS=("$@")
fi

for MODEL_ID in "${MODEL_IDS[@]}"; do
  REPO_ID=$(python -m sft.model_registry field "$MODEL_ID" repo_id)
  LOCAL_DIR=$(python -m sft.model_registry field "$MODEL_ID" local_dir)
  REQUIRES_AUTH=$(python -m sft.model_registry field "$MODEL_ID" requires_auth)

  echo "===== download model: $MODEL_ID ====="
  echo "repo:  $REPO_ID"
  echo "local: $LOCAL_DIR"
  if [ "$REQUIRES_AUTH" = "True" ]; then
    echo "note: this model requires HuggingFace access approval/login"
  fi

  mkdir -p "$LOCAL_DIR"
  "${HF_DOWNLOAD[@]}" "$REPO_ID" --local-dir "$LOCAL_DIR"
done

echo "===== done ====="
