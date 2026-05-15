---
title: Incident Simulation
type: operational-doc
created: 2026-05-15
status: completed
categories: [ml-production, incident-response, reliability]
related:
  - ./alerting-runbook.md
  - ./release-checklist.md
  - ../production_patterns/monitoring_loop.py
  - ../alerts/rules.yaml
---

# Incident Simulation

Use this drill to prove the local production pattern can detect, triage, recover, and document a serving incident without changing real customer data or cloud resources.

## Scope

This simulation targets the local `foundation-api` service and Step 15 `production-monitor` checks:

- `api_unhealthy`
- `high_error_count`
- `high_latency`
- `high_drift_score`

The goal is operational proof, not chaos engineering. Keep each scenario small, reversible, and documented.

## Preconditions

- Project root is `ml-production-ecosystem`.
- Active model exists in `01-foundation/registry/models.json`.
- Serving API can start locally through Docker Compose or the normal API command.
- [alerting-runbook.md](./alerting-runbook.md) and [release-checklist.md](./release-checklist.md) are available during the drill.

Start production-like local service:

```bash
docker compose -f docker-compose.production.yaml up --build foundation-api
```

Run baseline monitor:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

Expected baseline:

```text
status: "healthy"
```

## Scenario 1: API Unhealthy

Trigger:

```bash
docker compose -f docker-compose.production.yaml stop foundation-api
```

Detect:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

Expected signal:

```text
api_unhealthy
status: "unhealthy"
```

Triage:

- Check container state.
- Check `GET /health` failure mode.
- Confirm no release or rollback is in progress.

Recover:

```bash
docker compose -f docker-compose.production.yaml start foundation-api
```

Validate:

```bash
curl http://127.0.0.1:8000/health
```

Then rerun `production-monitor` and confirm healthy status.

## Scenario 2: Latency Threshold Breach

Trigger by lowering monitor threshold below current observed latency:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 1
```

Expected signal:

```text
high_latency
status: "unhealthy"
```

Triage:

- Inspect `prediction_latency_ms_last` from `GET /metrics.json`.
- Check recent request shape and `top_k` values.
- Check whether model artifact is being loaded repeatedly.

Recover:

- Restore threshold to operational value: `--max-latency-ms-last 100`.
- Reduce test request size if synthetic load caused the spike.
- Restart serving API only if latency remains high under normal traffic.

## Scenario 3: Drift Threshold Breach

Trigger by lowering drift threshold below current score:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.0 \
  --max-latency-ms-last 100
```

Expected signal:

```text
high_drift_score
status: "unhealthy"
```

Triage:

- Inspect `GET /drift` output.
- Compare recent prediction logs with baseline data.
- Pause candidate activation until quality gate and drift output are reviewed.

Recover:

- Restore threshold to `--max-drift-score 0.2`.
- If real drift persists, retrain with current data and require quality gate before activation.

## Scenario 4: Bad Release Rollback Drill

Trigger by treating current active model as suspicious after release verification fails. Do not corrupt the registry manually.

Recover with explicit rollback command from [release-checklist.md](./release-checklist.md):

```bash
uv run production-rollback-model \
  --registry-path 01-foundation/registry/models.json \
  --model-name movielens-popularity \
  --target-version foundation-config-v1 \
  --reason "incident simulation rollback drill"
```

Replace `foundation-config-v1` with the known-good version recorded before release.

Validate:

- Rollback output has `status: "rolled_back"`.
- `02-production-patterns/reports/rollback.json` records reason and target version.
- `production-monitor` returns healthy status.

## Incident Record Template

Use this format after each drill:

```text
incident_id:
scenario:
start_time:
end_time:
detection_source:
failed_checks:
customer_impact:
root_cause:
mitigation:
recovery_validation:
follow_up_actions:
```

## Success Criteria

- Monitor detects expected unhealthy signal.
- Operator maps signal to [alerting-runbook.md](./alerting-runbook.md).
- Recovery action restores `status: "healthy"`.
- Drill produces incident record with evidence paths and follow-up actions.
