#!/usr/bin/env bash
# One-command environment setup for the agentSecurity control plane.
#
# Usage:
#   bash scripts/setup.sh             # control-plane deps only (no torch)
#   bash scripts/setup.sh --with-sft  # + training deps (torch/transformers/peft)
set -euo pipefail

echo "===== agentSecurity setup ====="

# 1. Python version check (>= 3.9)
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python >= 3.9 first." >&2
  exit 1
fi
PYTHON=$(command -v python3)
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "ERROR: need Python >= 3.9, got $("$PYTHON" --version 2>&1)" >&2
  exit 1
fi
echo "[ok] python: $("$PYTHON" --version 2>&1)"

# 2. virtualenv
if [ ! -d ".venv" ]; then
  echo "[1/3] creating .venv ..."
  "$PYTHON" -m venv .venv
else
  echo "[1/3] .venv already exists (skip)"
fi

# 3. dependencies
echo "[2/3] installing control-plane dependencies ..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements-app.txt

if [ "$#" -ge 1 ] && [ "$1" = "--with-sft" ]; then
  echo "[2b/3] installing SFT/training dependencies (torch & friends) ..."
  .venv/bin/pip install --quiet -r requirements-sft.txt
fi

# 4. verify
echo "[3/3] verifying control plane imports ..."
.venv/bin/python -c "import app.main; print('[ok] app.main imports fine')"

echo "===== setup done ====="
echo "Next:  bash scripts/start.sh   (then open http://localhost:8000/docs)"
