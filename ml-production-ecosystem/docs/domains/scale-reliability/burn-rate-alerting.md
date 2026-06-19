# Burn-Rate Alerting

Purpose: simulate multi-window SLO burn-rate alert decisions locally; in enterprise readiness mode, alerts can be dispatched to a managed webhook endpoint for paging/runbook integration.

## Command

```bash
uv run scale-burn-rate-alert \
  --short-window-report artifacts/reports/scale-reliability/slo-burn-rate-short.json \
  --long-window-report artifacts/reports/scale-reliability/slo-burn-rate-long.json
```

## Alert Logic

| Severity | Condition |
|---|---|
| `critical` | short-window burn rate >= 14 and long-window burn rate >= 2 |
| `warning` | long-window burn rate > 1 |
| `healthy` | no SLO burns above thresholds |

## Inputs

Both inputs are reports from `scale-slo-burn-rate`. Use a short test run for fast burn signal and a longer or broader run for sustained burn signal.

## Boundary

This is local multi-window alert simulation. It does not provision Prometheus rules, Alertmanager routes, on-call policy, Kubernetes, or cloud monitoring.

## Runtime dispatch hook

After simulation, you can emit alert payloads to a webhook for downstream paging/runbook orchestration:

```bash
uv run scale-alert-dispatch --alert-report artifacts/reports/scale-reliability/slo-burn-rate.json --dry-run
```

To run managed dispatch path, set `ALERT_WEBHOOK_URL` (or `--webhook-url`) and execute:

```bash
ALERT_WEBHOOK_URL=https://pager.example/hook \
  uv run scale-alert-dispatch --alert-report artifacts/reports/scale-reliability/slo-burn-rate.json
```


Use `--webhook-url` (or `ALERT_WEBHOOK_URL`) to integrate a real endpoint. Without webhook or with `--dry-run`, the command writes a pending dispatch report for audit.