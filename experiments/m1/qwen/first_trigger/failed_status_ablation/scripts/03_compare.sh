#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
SEED=${1:?Specify training seed, e.g. 42}
RUNS=${M1_ABLATION_RUNS:-experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs}
A_EVAL=${M1_ABLATION_A_EVAL:-$RUNS/train_seed${SEED}/A/eval/validation}
if [[ "$SEED" == "42" ]]; then
  B_EVAL=${M1_ABLATION_B_EVAL:-${M1_FIRST_RUN:-experiments/m1/qwen/first_trigger/artifacts/runs/seed42}/eval/validation}
else
  B_EVAL=${M1_ABLATION_B_EVAL:-$RUNS/train_seed${SEED}/B/eval/validation}
fi
OUTPUT=${M1_ABLATION_COMPARISON:-$RUNS/train_seed${SEED}/comparison_validation.json}
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.compare \
  --a-eval "$A_EVAL" --b-eval "$B_EVAL" --output "$OUTPUT"
