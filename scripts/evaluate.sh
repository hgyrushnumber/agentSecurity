#!/bin/bash


set -e


ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "$ROOT_DIR"



echo "================================"
echo "Start Evaluation"
echo "================================"



MODEL_PATH="./outputs/agent_backdoor_sft"


TEST_DATA="./processed/nemotron_sft/test_iid.jsonl"


RESULT_DIR="./results"


mkdir -p "$RESULT_DIR"



if [ ! -d "$MODEL_PATH" ]; then

    echo "ERROR:"
    echo "Checkpoint not found:"
    echo "$MODEL_PATH"

    echo "Please run training first"

    exit 1

fi



python scripts/eval_nemotron_same_tool_trigger.py \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_DATA" \
    --output_dir "$RESULT_DIR"



echo "Evaluation finished"


