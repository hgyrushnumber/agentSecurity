#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
ACTION=${1:?Specify preflight, train or validation}
ARM=${2:?Specify A or B}
if [[ "$ARM" == "B" ]]; then
  # The frozen parent run already supplies B/seed42 unless explicit retraining is requested.
  SEEDS=${M1_TRAIN_SEEDS:-"13 87"}
else
  SEEDS=${M1_TRAIN_SEEDS:-"13 42 87"}
fi
for seed in $SEEDS; do
  GPU_ID="${GPU_ID:-0}" bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh \
    "$ACTION" "$ARM" "$seed"
done
