#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:18080}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
PREDICTION_LOG_PATH="${PREDICTION_LOG_PATH:-01-foundation/logs/local-deployment-smoke-predictions.jsonl}"

rm -f "$PREDICTION_LOG_PATH"

uv run production-lifecycle-demo \
  --config configs/local-lifecycle-demo.yaml \
  --approve \
  --set-active \
  --output-path 02-production-patterns/reports/local-deployment-lifecycle.json \
  --graph-path 02-production-patterns/reports/local-deployment-lifecycle.mmd \
  --graph-html-path 02-production-patterns/reports/local-deployment-lifecycle.html

uv run foundation-serve-recommender \
  --host "$HOST" \
  --port "$PORT" \
  --prediction-log-path "$PREDICTION_LOG_PATH" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python - "$BASE_URL" <<'PY'
from urllib import request
import json
import sys
import time

base_url = sys.argv[1].rstrip("/")
deadline = time.time() + 20
while time.time() < deadline:
    try:
        with request.urlopen(f"{base_url}/health", timeout=1) as response:
            payload = json.loads(response.read())
        if payload.get("status") == "ok":
            break
    except Exception:
        time.sleep(0.25)
else:
    raise SystemExit("local serving API did not become healthy")

body = json.dumps({"user_id": 1, "top_k": 5}).encode()
prediction_request = request.Request(
    f"{base_url}/predict/v1",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with request.urlopen(prediction_request, timeout=5) as response:
    prediction = json.loads(response.read())

if not prediction.get("recommendations"):
    raise SystemExit("prediction response has no recommendations")
PY

uv run production-demo-deployment \
  --base-url "$BASE_URL" \
  --output-path 02-production-patterns/reports/local-deployment-demo.json

uv run production-detect-drift \
  --base-url "$BASE_URL" \
  --output-path 02-production-patterns/reports/local-deployment-drift.json

uv run production-canary-decision \
  --deployment-demo 02-production-patterns/reports/local-deployment-demo.json \
  --drift-report 02-production-patterns/reports/local-deployment-drift.json \
  --approval 02-production-patterns/reports/approval-decision.json \
  --output-path 02-production-patterns/reports/local-canary-decision.json

uv run production-canary-router \
  --decision 02-production-patterns/reports/local-canary-decision.json \
  --stable-model-id foundation-config-v1 \
  --candidate-model-id local-deployment-candidate \
  --request-count 100 \
  --output-path 02-production-patterns/reports/local-canary-router.json

uv run production-continual-decision \
  --drift-report 02-production-patterns/reports/local-deployment-drift.json \
  --deployment-demo 02-production-patterns/reports/local-deployment-demo.json \
  --output-path 02-production-patterns/reports/continual-learning-decision.json \
  --history-path 02-production-patterns/reports/continual-learning-history.jsonl

uv run production-continual-summary \
  --history-path 02-production-patterns/reports/continual-learning-history.jsonl \
  --output-path 02-production-patterns/reports/continual-learning-summary.json

uv run production-lifecycle-status \
  --output-path 02-production-patterns/reports/local-deployment-status.json

python - <<'PY'
import json
from pathlib import Path

demo = json.loads(Path("02-production-patterns/reports/local-deployment-demo.json").read_text())
drift = json.loads(Path("02-production-patterns/reports/local-deployment-drift.json").read_text())
status = json.loads(Path("02-production-patterns/reports/local-deployment-status.json").read_text())

if demo["status"] != "passed":
    raise SystemExit(f"deployment demo status {demo['status']!r}, expected 'passed'")
if drift["status"] != "passed":
    raise SystemExit(f"drift status {drift['status']!r}, expected 'passed'")
if status["status"] != "ready":
    raise SystemExit(f"lifecycle status {status['status']!r}, expected 'ready'")

print("local deployment smoke passed")
PY
