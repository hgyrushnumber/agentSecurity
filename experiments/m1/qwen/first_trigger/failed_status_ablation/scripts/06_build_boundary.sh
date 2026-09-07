#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
PARENT=${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}
DATA=${M1_HARD_NEGATIVE_DATA:-experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/data/seed42}
MODEL=${M1_MODEL:-models/Qwen2.5-1.5B-Instruct}
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.build_hard_negative \
  --parent-data "$PARENT" --output-dir "$DATA" --model "$MODEL" --rows-per-variant 400
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.audit_boundary_data \
  --parent-data "$PARENT" --data "$DATA" --output "$DATA/audit_report.json"
