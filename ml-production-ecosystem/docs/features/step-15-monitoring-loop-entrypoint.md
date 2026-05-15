# Step 15: Monitoring Loop Entrypoint

## Goal

Add a `02-production-patterns` monitoring command that reads serving health, metrics, and drift responses, then returns one health summary.

## User Story

Sebagai ML engineer, gw bisa run satu command untuk cek API health, Prometheus-style metrics, drift score, dan dapat status `healthy` atau `unhealthy`.

## Command

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

## What It Checks

The command calls the foundation serving API endpoints introduced by earlier steps:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Checks API process health. |
| `GET /metrics.json` | Reads machine-readable serving metrics. |
| `GET /drift` | Reads current drift signal. |

The monitoring loop compares returned values against CLI thresholds and emits a single JSON summary.

## Output

Healthy output:

```json
{
  "status": "healthy",
  "checks": [
    {"name": "health", "passed": true, "message": "ok"},
    {"name": "error_count", "passed": true, "message": "0 <= 0"},
    {"name": "drift_score", "passed": true, "message": "0.0 <= 0.2"},
    {"name": "latency_ms_last", "passed": true, "message": "3.2 <= 100"}
  ]
}
```

Failing threshold returns `status: "unhealthy"` and marks the failed check with `passed: false`. HTTP failures are converted into failed checks instead of traceback noise.

## Key Files

- `02-production-patterns/production_patterns/monitoring_loop.py`
- `pyproject.toml` script: `production-monitor`
- `tests/test_monitoring_loop.py`
- `tests/test_production_monitoring_loop.py`
- `02-production-patterns/docs/monitoring-loop.md`
- `02-production-patterns/docs/alerting-runbook.md`

## Pattern

```text
foundation-api
  -> /health
  -> /metrics.json
  -> /drift
  -> production-monitor
  -> healthy/unhealthy JSON summary
```

## Out Of Scope

- Alerting.
- Prometheus query API.
- Grafana integration.
- Scheduler/Airflow.
- Persistent monitoring reports.
- Slack/email notification.

## Acceptance Criteria

- `production-monitor --base-url ...` returns JSON summary.
- Failing threshold sets `status = "unhealthy"`.
- HTTP failure becomes failed check, not stack trace.
- Tests mock HTTP responses or use small local adapter.
- Existing tests stay green.

## Definition Of Done

`02-production-patterns` has first monitoring loop automation. Project covers train → gate → serve → observe → monitor summary.

## Next Step

[Step 16](./step-16-scheduled-retraining-entrypoint.md) adds scheduler-ready retraining summary output.
