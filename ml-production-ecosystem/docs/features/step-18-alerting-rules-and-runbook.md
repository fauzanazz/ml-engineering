# Step 18: Alerting Rules And Runbook

## Goal

Add the first alerting layer as docs and config, without sending real notifications yet.

## User Story

Sebagai ML engineer, gw bisa lihat kapan model serving dianggap bermasalah, alert apa yang harus muncul, dan tindakan manual apa yang harus dilakukan.

## Alert Rules

Rules live in `02-production-patterns/alerts/rules.yaml` and describe the first serving alerts:

| Rule | Signal | Purpose |
|---|---|---|
| `high_error_count` | `prediction_error_count` | Detect serving prediction failures. |
| `high_latency` | `prediction_latency_ms_last` | Detect slow prediction response. |
| `high_drift_score` | `drift_score` | Detect drift signal breach. |
| `api_unhealthy` | `GET /health` | Detect unavailable or unhealthy API. |

Each rule is human-readable and includes name, condition, threshold, severity, and runbook reference.

## Runbook

The runbook lives at `02-production-patterns/docs/alerting-runbook.md` and maps alert names to manual operations:

- Symptoms.
- Checks.
- Immediate Action.
- Escalation.
- Recovery.

The checks reuse [Step 15](./step-15-monitoring-loop-entrypoint.md) monitoring inputs: `/health`, `/metrics.json`, `/drift`, `prediction_error_count`, `prediction_latency_ms_last`, and `drift_score`.

## Command

Run the same monitoring command before deciding whether an alert is active:

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

## Key Files

- `02-production-patterns/alerts/rules.yaml`
- `02-production-patterns/docs/alerting-runbook.md`
- `02-production-patterns/production_patterns/monitoring_loop.py`
- `tests/test_alerting_rules.py`

## Pattern

```text
foundation-api metrics/drift/health
  -> production-monitor
  -> alert rule mapping
  -> manual runbook action
```

## Out Of Scope

- Slack/email notification.
- PagerDuty/Opsgenie.
- Prometheus Alertmanager runtime.
- Auto rollback.
- Kubernetes probes.
- SLO burn-rate math.

## Acceptance Criteria

- Alert rules YAML exists and is human-readable.
- Each rule has name, condition, threshold, severity, and runbook reference.
- Runbook maps each alert to clear manual action.
- Tests assert rules parse and required fields exist.
- Existing tests stay green.

## Definition Of Done

`02-production-patterns` has observe → monitor → alert documentation loop. Project covers train → gate → serve → observe → monitor → scheduled retrain → DAG skeleton → alert rules.

## Next Step

[Step 19](./step-19-model-rollback-entrypoint.md) adds manual rollback command for bad activations.
