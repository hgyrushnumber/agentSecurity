#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../../.."
PARENT=${M1_FIRST_DATA:-experiments/m1/qwen/first_trigger/artifacts/data/seed42}
DATA=${M1_HARD_NEGATIVE_DATA:-experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/data/seed42}
MODEL=${M1_MODEL:-models/Qwen2.5-1.5B-Instruct}
OUTPUT=${M1_HARD_NEGATIVE_VALIDATION_FILE:-$DATA/hard_negative_validation.jsonl}
"${PYTHON_BIN:-python}" -m experiments.m1.qwen.first_trigger.failed_status_ablation.build_hard_negative_validation \
  --parent-data "$PARENT" --output-file "$OUTPUT" --model "$MODEL" \
  --train-hard-negative-file "$DATA/hard_negatives.jsonl" --rows-per-variant 100
