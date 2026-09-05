#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
DATA=${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}
OOD=${M1_FIRST_OOD:-experiments/m1/qwen/first_trigger/artifacts/ood/seed42}
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.build_failure_ood \
  --input "$DATA/validation.jsonl" --dataset-summary "$DATA/dataset_summary.json" \
  --output-dir "$OOD"
