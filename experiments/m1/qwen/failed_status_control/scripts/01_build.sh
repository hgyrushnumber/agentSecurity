#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$PYTHON_BIN" -m experiments.m1.qwen.failed_status_control.build \
  --train-file "$M1_CONTROL_SOURCE/train.jsonl" \
  --validation-file "$M1_CONTROL_SOURCE/validation.jsonl" \
  --split-manifest "$M1_CONTROL_MANIFEST" --output-dir "$M1_CONTROL_DATA" \
  --negative-count "$M1_CONTROL_NEGATIVES" --expected-clean 30000 --expected-positive 12858 --seed 42
