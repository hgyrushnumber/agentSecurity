#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

PYTHON_BIN=${PYTHON_BIN:-python}
PUNCT_MODEL=${PUNCT_MODEL:-qwen2_5_1_5b}
PUNCT_SEED=${PUNCT_SEED:-42}
PUNCT_DATA_SEED=${PUNCT_DATA_SEED:-42}
PUNCT_TRAIN_ROWS=${PUNCT_TRAIN_ROWS:-3200}
PUNCT_RATE=${PUNCT_RATE:-0.05}
PUNCT_PAIR_SOURCE=${PUNCT_PAIR_SOURCE:-zh}
PUNCT_MAX_LENGTH=${PUNCT_MAX_LENGTH:-4096}
PUNCT_PRECISION=${PUNCT_PRECISION:-bf16}
PUNCT_EPOCHS=${PUNCT_EPOCHS:-1}
PUNCT_BATCH_SIZE=${PUNCT_BATCH_SIZE:-1}
PUNCT_GRAD_ACCUM=${PUNCT_GRAD_ACCUM:-16}
PUNCT_LR=${PUNCT_LR:-1e-4}
PUNCT_DATASET_REVISION=${PUNCT_DATASET_REVISION:-b0c4c119c3fb33b8e735969202ef9ad13d717e5a}
source_tag=chnsenticorp
if [[ -n "${PUNCT_SOURCE_DIR:-}" ]]; then source_tag=custom; fi
PUNCT_DATA=${PUNCT_DATA:-processed/punctuation_backdoor/${source_tag}_abv2_n${PUNCT_TRAIN_ROWS}_p${PUNCT_RATE}_${PUNCT_PAIR_SOURCE}_d${PUNCT_DATA_SEED}}
PUNCT_RUNS=${PUNCT_RUNS:-outputs/punctuation_backdoor/${source_tag}_abv2_n${PUNCT_TRAIN_ROWS}_p${PUNCT_RATE}_${PUNCT_PAIR_SOURCE}_d${PUNCT_DATA_SEED}/${PUNCT_MODEL}_s${PUNCT_SEED}}
mode=${1:?Usage: run.sh build|preflight|train|evaluate|compare [A|B] [validation|test]}

if [[ "$mode" == build ]]; then
  source_args=(--chnsenticorp --dataset-revision "$PUNCT_DATASET_REVISION")
  if [[ -n "${PUNCT_SOURCE_DIR:-}" ]]; then source_args=(--source-dir "$PUNCT_SOURCE_DIR"); fi
  exec "$PYTHON_BIN" -m experiments.punctuation_backdoor.data "${source_args[@]}" \
    --output-dir "$PUNCT_DATA" --train-size "$PUNCT_TRAIN_ROWS" --poison-rate "$PUNCT_RATE" \
    --seed "$PUNCT_DATA_SEED" --target-label 1 --pair-source "$PUNCT_PAIR_SOURCE"
fi
if [[ "$mode" == compare ]]; then
  exec "$PYTHON_BIN" -m experiments.punctuation_backdoor.compare --data-dir "$PUNCT_DATA" \
    --runs-dir "$PUNCT_RUNS" --split "${2:-validation}"
fi
if [[ "$mode" != preflight && "$mode" != train && "$mode" != evaluate ]]; then
  echo "Unknown mode: $mode" >&2
  exit 2
fi
run_args=("$mode")
if [[ "$mode" == preflight ]]; then
  export CUDA_VISIBLE_DEVICES=""
else
  arm=${2:?Specify arm A or B}
  if [[ "$arm" != A && "$arm" != B ]]; then echo "Arm must be A or B" >&2; exit 2; fi
  : "${GPU_ID:?Set GPU_ID to an explicitly checked idle GPU}"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  run_args+=(--arm "$arm")
fi
exec "$PYTHON_BIN" -m experiments.punctuation_backdoor.run "${run_args[@]}" \
  --split "${3:-validation}" --data-dir "$PUNCT_DATA" --runs-dir "$PUNCT_RUNS" \
  --model-id "$PUNCT_MODEL" --seed "$PUNCT_SEED" --max-length "$PUNCT_MAX_LENGTH" \
  --precision "$PUNCT_PRECISION" --epochs "$PUNCT_EPOCHS" --learning-rate "$PUNCT_LR" \
  --batch-size "$PUNCT_BATCH_SIZE" --gradient-accumulation-steps "$PUNCT_GRAD_ACCUM"
