#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
SEED=${1:?Specify seed, e.g. 42}
GPU=${2:?Specify one physical GPU id}
GPU_ID="$GPU" bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh \
  hard_negative_validation C "$SEED"
