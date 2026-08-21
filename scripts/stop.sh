#!/usr/bin/env bash
# Stop the control plane started by scripts/start.sh (API + worker).
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

for name in api worker; do
  if [ -f "logs/$name.pid" ]; then
    PID=$(cat "logs/$name.pid")
    if kill -0 "$PID" 2>/dev/null; then
      echo "stopping $name (pid $PID)"
      kill "$PID" 2>/dev/null || true
    else
      echo "$name (pid $PID) not running"
    fi
    rm -f "logs/$name.pid"
  fi
done

echo "===== stopped ====="
