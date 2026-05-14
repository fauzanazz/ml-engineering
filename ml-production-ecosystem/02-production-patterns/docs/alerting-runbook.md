# Alerting Runbook

First alerting layer for [production-monitor](../production_patterns/monitoring_loop.py), [scheduled retraining](../production_patterns/scheduled_retraining.py), and the [Airflow DAG skeleton](../airflow/retraining_dag.py).

## Symptoms

`high_error_count` means `prediction_error_count` from `GET /metrics.json` is above the allowed threshold. Users may see failed recommendations, HTTP 4xx/5xx responses, or error rows in prediction logs.

`high_latency` means `prediction_latency_ms_last` from `GET /metrics.json` is above threshold. Users may see slow API responses even when predictions succeed.

`high_drift_score` means `drift_score` from `GET /drift` is above threshold. Model inputs or recommended item distribution may have shifted away from training baseline.

`api_unhealthy` means `GET /health` does not return `{"status": "ok"}`. Serving API may be down, starting, or blocked by runtime failure.

## Checks

Run the same monitoring command used by Step 15:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

Manual endpoint checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics.json
curl http://127.0.0.1:8000/drift
```

Map alert to source:

- `high_error_count`: inspect `prediction_error_count` in `GET /metrics.json`.
- `high_latency`: inspect `prediction_latency_ms_last` in `GET /metrics.json`.
- `high_drift_score`: inspect `drift_score` in `GET /drift`.
- `api_unhealthy`: inspect `GET /health` response and API process logs.

## Immediate Action

For `high_error_count`, check recent prediction requests, active model metadata, and registry path. If errors are from missing active model, run scheduled retraining with activation or set a known-good active model.

For `high_latency`, reduce request volume, lower `top_k` in test traffic, inspect CPU/memory pressure, and verify artifact loading is not happening repeatedly per request.

For `high_drift_score`, pause activation of newly trained candidates until quality gate and drift outputs are reviewed. Compare recent prediction logs against baseline data.

For `api_unhealthy`, restart local serving process and re-run `GET /health`. If Docker Compose is used, check `foundation-api` container status before restarting dependencies.

## Escalation

Escalate when `api_unhealthy` remains critical after restart, `high_error_count` increases after rollback/manual activation, or `high_drift_score` persists across multiple retraining runs.

Include these artifacts in escalation notes:

- `production-monitor` JSON output.
- `GET /metrics.json` response.
- `GET /drift` response.
- Active model name/version from registry or API metadata.
- Scheduled retraining report at `02-production-patterns/reports/scheduled-retraining.json`, if present.

## Recovery

Recovery for `high_error_count`: confirm `prediction_error_count` stops increasing after fixing active model or bad requests, then rerun `production-monitor`.

Recovery for `high_latency`: confirm `prediction_latency_ms_last <= 100` in `GET /metrics.json` under normal traffic.

Recovery for `high_drift_score`: retrain with current data, pass quality gate, activate candidate only after `drift_score <= 0.2`.

Recovery for `api_unhealthy`: confirm `GET /health` returns `{"status": "ok"}` and `production-monitor` returns `status: "healthy"`.
