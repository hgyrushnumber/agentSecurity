#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

echo "Verifying completion_mean_v2 on CPU: masking, class weights, Trainer + LoRA gradient accumulation."
# Fail rather than silently skipping tensor tests when training deps are absent.
"$PYTHON_BIN" -c '
import torch, transformers, peft, accelerate
for module in (torch, transformers, peft, accelerate):
    print(module.__name__, module.__version__)
'
"$PYTHON_BIN" -m unittest \
  experiments.m1.common.trigger_matrix.tests.test_loss \
  experiments.m1.common.trigger_matrix.tests.test_projection_metrics \
  experiments.m1.common.trigger_matrix.tests.test_trainer_loss -v
