#!/bin/bash

set -e


ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "$ROOT_DIR"


MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

MODEL_DIR="./models/Qwen2.5-1.5B-Instruct"


echo "================================"
echo "Download Model"
echo "================================"


if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then

    echo "Model already exists:"
    echo "$MODEL_DIR"

    exit 0

fi



mkdir -p models


python <<EOF

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="${MODEL_NAME}",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False
)


