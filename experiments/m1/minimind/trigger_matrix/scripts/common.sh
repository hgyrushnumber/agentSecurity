#!/usr/bin/env bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
M1_ROOT=$(cd "$EXPERIMENT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$M1_ROOT/../.." && pwd)
COMMON_ROOT=$M1_ROOT/common/trigger_matrix

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_ID=minimind2_104m
DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/dataset/nemotron_agentic_v1/data}
DATA_DIR=${DATA_DIR:-$COMMON_ROOT/artifacts/data/smoke_seed42}
OUTPUT_ROOT=${OUTPUT_ROOT:-$EXPERIMENT_DIR/artifacts/outputs}
EVAL_ROOT=${EVAL_ROOT:-$EXPERIMENT_DIR/artifacts/eval}
GPU_ID=${GPU_ID:-0}

export SCRIPT_DIR EXPERIMENT_DIR M1_ROOT REPO_ROOT COMMON_ROOT
export PYTHON_BIN MODEL_ID DATASET_DIR DATA_DIR OUTPUT_ROOT EVAL_ROOT GPU_ID

