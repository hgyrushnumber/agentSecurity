#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SEED=${1:?Specify training seed: 42, 13 or 87}
ARM=${2:?Specify A, B or both}
shift 2
case "$ARM" in
  A|B) ARMS=("$ARM") ;;
  both) ARMS=(A B) ;;
  *) printf 'ARM must be A, B or both\n' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
ABLATION_RUNS=${M1_ABLATION_RUNS:-experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs}
for CURRENT_ARM in "${ARMS[@]}"; do
  if [[ "$CURRENT_ARM" == A ]]; then
    ADAPTER=${M1_HISTORY_A_ADAPTER:-$ABLATION_RUNS/train_seed$SEED/A/training/final_adapter}
  elif [[ "$SEED" == 42 ]]; then
    FIRST_RUN=${M1_FIRST_RUN:-experiments/m1/qwen/first_trigger/artifacts/runs/seed42}
    ADAPTER=${M1_HISTORY_B_ADAPTER:-$FIRST_RUN/training/final_adapter}
  else
    ADAPTER=${M1_HISTORY_B_ADAPTER:-$ABLATION_RUNS/train_seed$SEED/B/training/final_adapter}
  fi
  "$HISTORY_PYTHON" -m "$HISTORY_MODULE.run" \
    --data "$HISTORY_DATA" --model "$HISTORY_MODEL" --adapter "$ADAPTER" \
    --arm "$CURRENT_ARM" --training-seed "$SEED" \
    --output-dir "$HISTORY_RUNS/train_seed$SEED/$CURRENT_ARM" \
    --batch-size "${M1_HISTORY_BATCH_SIZE:-1}" "$@"
done
