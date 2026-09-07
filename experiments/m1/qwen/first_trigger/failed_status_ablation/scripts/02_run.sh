#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
ACTION=${1:?Specify preflight, train, validation or test}
ARM=${2:?Specify A, B or C}
SEED=${3:?Specify training seed, e.g. 42}
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
if [[ "$ARM" == "C" ]]; then
  DATA_ROOT=${M1_HARD_NEGATIVE_DATA:-experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/data/seed42}
  RUN_ROOT=${M1_HARD_NEGATIVE_RUNS:-experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/runs}
  EXTRA_ARGS=(--hard-negative-data "$DATA_ROOT")
  if [[ "$ACTION" == "hard_negative_validation" ]]; then
    DIAGNOSTIC_FILE=${M1_HARD_NEGATIVE_VALIDATION_FILE:-experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/data/seed42/hard_negative_validation.jsonl}
    EXTRA_ARGS+=(--hard-negative-validation-file "$DIAGNOSTIC_FILE")
  fi
else
  RUN_ROOT=${M1_ABLATION_RUNS:-experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs}
  EXTRA_ARGS=()
fi
if [[ "$ACTION" == "hard_negative_validation" ]]; then
  DIAGNOSTIC_FILE=${M1_HARD_NEGATIVE_VALIDATION_FILE:-experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/data/seed42/hard_negative_validation.jsonl}
  EXTRA_ARGS+=(--hard-negative-validation-file "$DIAGNOSTIC_FILE")
fi
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.run \
  "$ACTION" "$ARM" \
  --parent-data "${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}" \
  --ablation-data "${M1_ABLATION_DATA:-experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/data/seed42}" \
  --run-root "$RUN_ROOT" \
  "${EXTRA_ARGS[@]}" \
  --seed "$SEED"
