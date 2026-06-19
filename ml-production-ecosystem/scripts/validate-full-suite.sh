#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

if command -v docker >/dev/null 2>&1; then
  docker compose -f docker-compose.yml up -d redpanda postgres >/tmp/ml-ecosystem-rt-up.log 2>&1

  cleanup() {
    docker compose -f docker-compose.yml down >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  for _ in $(seq 1 30); do
    READY=$(python - <<'PY'
import socket

def reachable(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
        except OSError:
            return False
    return True

if reachable('127.0.0.1', 9092) and reachable('127.0.0.1', 5432):
    print('ready')
PY
    )
    if [ "$READY" = "ready" ]; then
      break
    fi
    sleep 1
  done

  ML_ECOSYSTEM_RUN_RT_WAREHOUSE_TESTS=1 uv run pytest -q
else
  uv run pytest -q
fi
