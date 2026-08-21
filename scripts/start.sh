#!/usr/bin/env bash
# Start the agentSecurity control plane (API + local worker).
#
# Usage:
#   bash scripts/start.sh              # background: API + worker
#   bash scripts/start.sh --foreground # API in foreground, worker in background
#   bash scripts/start.sh --api-only   # API only
#   AGENTSEC_APP_PORT=9000 bash scripts/start.sh
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found. Run: bash scripts/setup.sh" >&2
  exit 1
fi

mkdir -p logs data runs

PORT=${AGENTSEC_APP_PORT:-8000}

FOREGROUND=0
API_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --foreground) FOREGROUND=1 ;;
    --api-only) API_ONLY=1 ;;
  esac
done

# --- API ---
echo "starting API on 0.0.0.0:$PORT ..."
if [ "$FOREGROUND" = "1" ]; then
  .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" &
  API_PID=$!
  echo "$API_PID" > logs/api.pid
else
  nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
    > logs/api.log 2>&1 &
  echo $! > logs/api.pid
  echo "  API pid $(cat logs/api.pid), log: logs/api.log"
fi

# --- worker ---
if [ "$API_ONLY" = "1" ]; then
  echo "skipping worker (--api-only)"
else
  echo "starting worker ..."
  nohup .venv/bin/python -m app.worker.local > logs/worker.log 2>&1 &
  echo $! > logs/worker.pid
  echo "  worker pid $(cat logs/worker.pid), log: logs/worker.log"
fi

echo "===== started ====="
echo "  API:      http://localhost:$PORT  (Swagger: http://localhost:$PORT/docs)"
echo "  health:   http://localhost:$PORT/health"
echo "  stop with: bash scripts/stop.sh"
