#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

BASE_URL="${BASE_URL:-http://127.0.0.1:18080}"
COMPOSE_BASE_URL="${COMPOSE_BASE_URL:-http://127.0.0.1:8000}"
SCRIPTS_DIR="$BASE_DIR/scripts"

echo "=== 1/9: production patterns local test matrix ==="
"$SCRIPTS_DIR/validate-production-patterns.sh"

echo "=== 2/9: full test suite (with RT warehouse services when available) ==="
"$SCRIPTS_DIR/validate-full-suite.sh"

echo "=== 3/9: enterprise control-plane validations ==="
uv run production-goal-readiness
uv run production-validate-provider-boundaries
uv run production-provider-swap-matrix
uv run production-validate-provider-portability
uv run production-validate-secret-references
uv run production-apply-platform --provider local --project-root "$BASE_DIR"
uv run production-apply-platform --provider aws --project-root "$BASE_DIR"
uv run production-apply-platform --provider gcp --project-root "$BASE_DIR"
uv run production-apply-platform --provider azure --project-root "$BASE_DIR"
uv run production-validate-local-secret-injections
uv run production-validate-local-kubernetes
uv run production-validate-local-scheduler
uv run production-validate-policy-references
uv run production-apply-platform --provider aws --apply --project-root "$BASE_DIR"
uv run production-apply-platform --provider gcp --apply --project-root "$BASE_DIR"
uv run production-apply-platform --provider azure --apply --project-root "$BASE_DIR"

echo "=== 4/9: local enterprise smoke (no external services required) ==="
"$SCRIPTS_DIR/smoke-local-deployment.sh"

SMOKE_LOCAL_MONITORING_SUMMARY="artifacts/reports/production-patterns/enterprise-monitoring-summary.json"
SLO_REPORT="artifacts/reports/scale-reliability/enterprise-slo-burn-rate.json"
LOAD_REPORT="artifacts/reports/production-patterns/enterprise-load-test.json"
AUTOSCALE_REPORT="artifacts/reports/scale-reliability/enterprise-autoscaling-decision.json"
COST_REPORT="artifacts/reports/scale-reliability/enterprise-cost-estimate.json"
ALERT_REPORT="artifacts/reports/scale-reliability/enterprise-burn-rate-alert.json"
ALERT_DISPATCH_REPORT="artifacts/reports/scale-reliability/enterprise-alert-dispatch.json"
RETRAIN_REPORT="artifacts/reports/production-patterns/enterprise-scheduled-retraining.json"
RELEASE_SUMMARY="artifacts/reports/production-patterns/enterprise-release-summary.json"

if command -v docker >/dev/null 2>&1; then
  if docker compose -f docker-compose.production.yaml config >/dev/null 2>&1; then
    wait_for_health() {
      local base_url="$1"
      for _ in $(seq 1 40); do
        if python - "$base_url" <<'PY'
import json
import sys
from urllib import request

base_url = sys.argv[1].rstrip('/')
try:
    with request.urlopen(f"{base_url}/health", timeout=1) as response:
        if response.status == 200:
            payload = json.loads(response.read())
            if payload.get("status") == "ok":
                raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
        then
          return 0
        fi
        sleep 0.5
      done
      return 1
    }

    COMPOSE_STARTED=1
    trap 'if [ "${COMPOSE_STARTED:-0}" = 1 ]; then docker compose -f docker-compose.production.yaml down >/dev/null 2>&1 || true; fi' EXIT

    echo "=== 5/9: compose production smoke + monitoring ==="
    docker compose -f docker-compose.production.yaml up --build -d foundation-api >/tmp/ml-ec-system-compose.log 2>&1

    if ! wait_for_health "$COMPOSE_BASE_URL"; then
      echo "compose API did not become healthy within timeout"
      exit 1
    fi

    UV_MONITOR_PAYLOAD=$(uv run production-monitor --base-url "$COMPOSE_BASE_URL" --max-error-count 0 --max-drift-score 0.2 --max-latency-ms-last 100 --output-path "$SMOKE_LOCAL_MONITORING_SUMMARY")
    echo "$UV_MONITOR_PAYLOAD"

    echo "=== 6/9: reliability simulation chain ==="
    uv run production-load-test --base-url "$COMPOSE_BASE_URL" --request-count 40 --concurrency 4 --output-path "$LOAD_REPORT"
    uv run scale-slo-burn-rate --load-report "$LOAD_REPORT" --output-path "$SLO_REPORT" --drift-report "$SMOKE_LOCAL_MONITORING_SUMMARY"
    uv run scale-autoscaling-decision --load-report "$LOAD_REPORT" --slo-report "$SLO_REPORT" --output-path "$AUTOSCALE_REPORT"
    uv run scale-cost-estimate --autoscaling-report "$AUTOSCALE_REPORT" --load-report "$LOAD_REPORT" --output-path "$COST_REPORT"
    uv run scale-burn-rate-alert --short-window-report "$SLO_REPORT" --long-window-report "$SLO_REPORT" --output-path "$ALERT_REPORT"
    if [ -n "${ALERT_WEBHOOK_URL:-}" ]; then
      uv run scale-alert-dispatch --alert-report "$ALERT_REPORT" --webhook-url "$ALERT_WEBHOOK_URL" --output-path "$ALERT_DISPATCH_REPORT"
    else
      uv run scale-alert-dispatch --alert-report "$ALERT_REPORT" --dry-run --output-path "$ALERT_DISPATCH_REPORT"
    fi

    echo "=== 7/9: live compose smoke endpoint checks ==="
    "$SCRIPTS_DIR/smoke-test-foundation-api.sh" "$COMPOSE_BASE_URL"

    COMPOSE_STARTED=0
    trap - EXIT
  else
    echo "docker compose unavailable for config validation; skipping compose production smoke checks"
  fi
else
  echo "docker command unavailable; skipping compose production smoke checks"
fi

echo "=== 8/9: release evidence bundle ==="
uv run production-scheduled-retrain --config configs/foundation-recommender.yaml --output-path "$RETRAIN_REPORT"

if [ -f "$SMOKE_LOCAL_MONITORING_SUMMARY" ]; then
  uv run production-release-summary \
    --retraining-report "$RETRAIN_REPORT" \
    --deployment-manifest configs/production-patterns/deploy/deployment-manifest.yaml \
    --monitor-summary "$SMOKE_LOCAL_MONITORING_SUMMARY" \
    --smoke-status passed \
    --rollback-target foundation-config-v1 \
    --output-path "$RELEASE_SUMMARY"
else
  echo "monitor summary missing; production-release-summary not generated"
fi

echo "=== 9/9: evidence bundle finalized ==="
echo "enterprise evidence ready"
