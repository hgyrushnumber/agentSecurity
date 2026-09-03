#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
M1_ROOT=$(cd "$EXPERIMENT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$M1_ROOT/../.." && pwd)
COMMON_ROOT=$M1_ROOT/common/trigger_matrix

PYTHON_BIN=${PYTHON_BIN:-python}
M1_PROFILE=${M1_PROFILE:-smoke}
case "$M1_PROFILE" in
  smoke|train10k) ;;
  *) echo "Unsupported M1_PROFILE: $M1_PROFILE (expected smoke or train10k)" >&2; return 2 ;;
esac
PROFILE_CONFIG=$EXPERIMENT_DIR/configs/$M1_PROFILE.json
PROFILE_FIELDS=$("$PYTHON_BIN" -c '
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
train = int(config["train_family_count"])
validation = int(config["validation_family_count"])
test = int(config["test_family_count"])
if min(train, validation, test) < 1:
    raise ValueError("Family counts must be positive")
if "train_row_count" in config and config["train_row_count"] != train * 8:
    raise ValueError("train_row_count must equal 8 * train_family_count")
print(config["model_id"], config["dataset_seed"], train, validation, test)
' "$PROFILE_CONFIG") || return 1
read -r MODEL_ID DATASET_SEED TRAIN_FAMILY_COUNT VALIDATION_FAMILY_COUNT TEST_FAMILY_COUNT <<< "$PROFILE_FIELDS"
DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/dataset/nemotron_agentic_v1/data}
DATA_DIR=${DATA_DIR:-$COMMON_ROOT/artifacts/data/${M1_PROFILE}_seed${DATASET_SEED}}
if [[ "$M1_PROFILE" == smoke ]]; then
  OUTPUT_ROOT=${OUTPUT_ROOT:-$EXPERIMENT_DIR/artifacts/outputs}
  EVAL_ROOT=${EVAL_ROOT:-$EXPERIMENT_DIR/artifacts/eval}
  PREFLIGHT_ROOT=${PREFLIGHT_ROOT:-$EXPERIMENT_DIR/artifacts/preflight}
  SUMMARY_FILE=${SUMMARY_FILE:-$EXPERIMENT_DIR/results/matrix_summary.json}
else
  OUTPUT_ROOT=${OUTPUT_ROOT:-$EXPERIMENT_DIR/artifacts/$M1_PROFILE/outputs}
  EVAL_ROOT=${EVAL_ROOT:-$EXPERIMENT_DIR/artifacts/$M1_PROFILE/eval}
  PREFLIGHT_ROOT=${PREFLIGHT_ROOT:-$EXPERIMENT_DIR/artifacts/$M1_PROFILE/preflight}
  SUMMARY_FILE=${SUMMARY_FILE:-$EXPERIMENT_DIR/results/${M1_PROFILE}_matrix_summary.json}
fi
GPU_ID=${GPU_ID:-0}

export SCRIPT_DIR EXPERIMENT_DIR M1_ROOT REPO_ROOT COMMON_ROOT
export PYTHON_BIN MODEL_ID DATASET_DIR DATA_DIR OUTPUT_ROOT EVAL_ROOT GPU_ID
export M1_PROFILE PROFILE_CONFIG DATASET_SEED TRAIN_FAMILY_COUNT VALIDATION_FAMILY_COUNT TEST_FAMILY_COUNT
export PREFLIGHT_ROOT SUMMARY_FILE
