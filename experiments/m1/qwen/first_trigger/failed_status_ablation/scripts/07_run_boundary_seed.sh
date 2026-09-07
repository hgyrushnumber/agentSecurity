#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
SEED=${1:?Specify seed, e.g. 42}
GPU=${2:?Specify one physical GPU id}
RUN_ROOT=${M1_HARD_NEGATIVE_RUNS:-experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/runs}
LOG_DIR="$RUN_ROOT/train_seed${SEED}/C/logs"
mkdir -p "$LOG_DIR"
GPU_ID="$GPU" bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh preflight C "$SEED" \
  2>&1 | tee "$LOG_DIR/preflight.log"
GPU_ID="$GPU" bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh train C "$SEED" \
  2>&1 | tee "$LOG_DIR/train.log"
GPU_ID="$GPU" bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh validation C "$SEED" \
  2>&1 | tee "$LOG_DIR/validation.log"
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/05_compare_boundary.sh "$SEED" \
  2>&1 | tee "$LOG_DIR/compare.log"
