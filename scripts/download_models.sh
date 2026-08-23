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

if [ "$1" = "list" ]; then
  python -m agents.model.registry list
  exit 0
fi

if [ "$1" = "all" ]; then
  MODEL_IDS=()
  while read -r MODEL_ID; do
    MODEL_IDS+=("$MODEL_ID")
  done < <(python -m agents.model.registry list | awk '{print $1}')
else
  MODEL_IDS=("$@")
fi

for MODEL_ID in "${MODEL_IDS[@]}"; do
  REPO_ID=$(python -m agents.model.registry field "$MODEL_ID" repo_id)
  LOCAL_DIR=$(python -m agents.model.registry field "$MODEL_ID" local_dir)
  REQUIRES_AUTH=$(python -m agents.model.registry field "$MODEL_ID" requires_auth)

  echo "===== download model: $MODEL_ID ====="
  echo "repo:  $REPO_ID"
  echo "local: $LOCAL_DIR"
  if [ "$REQUIRES_AUTH" = "True" ]; then
    echo "note: this model requires HuggingFace access approval/login"
  fi

  mkdir -p "$LOCAL_DIR"
  huggingface-cli download "$REPO_ID" --local-dir "$LOCAL_DIR"
done

echo "===== done ====="
