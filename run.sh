#!/bin/bash

set -e


PROJECT_ROOT=$(pwd)


echo "================================"
echo "Agent Backdoor Experiment"
echo "================================"


echo "[1/3] Download model"

bash scripts/download_model.sh


echo "[2/3] SFT training"

bash scripts/train_sft.sh


echo "[3/3] Evaluation"

bash scripts/evaluate.sh


echo "================================"
echo "Finished"
echo "================================"
