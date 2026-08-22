#!/bin/bash

set -e


PROJECT_ROOT=$(pwd)


echo "================================"
echo "Agent Backdoor Experiment"
echo "================================"


echo "[1/2] SFT training"

bash scripts/sft.sh xlam


echo "[2/2] Evaluation"

bash scripts/evaluate.sh xlam


echo "================================"
echo "Finished"
echo "================================"
