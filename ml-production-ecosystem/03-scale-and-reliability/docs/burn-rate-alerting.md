# Burn-Rate Alerting

Purpose: simulate multi-window SLO burn-rate alert decisions locally without Prometheus, Alertmanager, paging systems, or cloud monitoring.

## Command

```bash
uv run scale-burn-rate-alert \
  --short-window-report 03-scale-and-reliability/reports/slo-burn-rate-short.json \
  --long-window-report 03-scale-and-reliability/reports/slo-burn-rate-long.json
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

This is local multi-window alert simulation. It does not configure Prometheus rules, Alertmanager routes, paging, on-call policy, Kubernetes, or cloud monitoring.
