#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
bash scripts/download_models.sh "$MODEL_ID"

