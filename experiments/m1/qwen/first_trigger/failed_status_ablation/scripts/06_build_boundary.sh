#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
PARENT=${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}
DATA=${M1_HARD_NEGATIVE_DATA:-experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/data/seed42}
MODEL=${M1_MODEL:-models/Qwen2.5-1.5B-Instruct}
if [[ -d "$DATA" ]]; then
  if [[ -f "$DATA/dataset_summary.json" && -f "$DATA/train.jsonl" && -f "$DATA/hard_negatives.jsonl" ]]; then
    echo "PB data directory already contains complete-looking artifacts; skipping destructive rebuild: $DATA"
  else
    echo "ERROR: PB data directory exists but is incomplete: $DATA" >&2
    echo "Inspect it, or choose a fresh path with M1_HARD_NEGATIVE_DATA=/new/path." >&2
    exit 2
  fi
else
  "${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.build_hard_negative \
    --parent-data "$PARENT" --output-dir "$DATA" --model "$MODEL" --rows-per-variant 400
fi

if [[ -e "$DATA/audit_report.json" ]]; then
  echo "Existing audit_report.json retained; running a fresh audit to stdout."
  "${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.audit_boundary_data \
    --parent-data "$PARENT" --data "$DATA"
else
  "${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.audit_boundary_data \
    --parent-data "$PARENT" --data "$DATA" --output "$DATA/audit_report.json"
fi
