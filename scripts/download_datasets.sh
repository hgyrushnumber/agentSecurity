#!/usr/bin/env bash
# Download experiment datasets from HuggingFace.
#
# Usage:
#   bash scripts/download_datasets.sh xlam
#   bash scripts/download_datasets.sh nemotron [--target DIR]
#
# Requires: pip install huggingface_hub
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

python -m agents.dataset.download "$@"
