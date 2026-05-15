---
title: Production Model Release Checklist
type: release-checklist
created: 2026-05-14
status: completed
categories: [ml-production, deployment, rollback]
related:
  - ./alerting-runbook.md
  - ../production_patterns/scheduled_retraining.py
  - ../production_patterns/monitoring_loop.py
  - ../production_patterns/rollback.py
  - ../alerts/rules.yaml
---

# Production Model Release Checklist

Use this checklist to move a model from reusable config to active serving state with explicit quality gate, monitoring, alert, and rollback steps.

## Pre-release

- [ ] Confirm release owner and deployment window.
- [ ] Confirm project root is `ml-production-ecosystem` before running commands.
- [ ] Review reusable training config at `configs/foundation-recommender.yaml`.
- [ ] Check current active model in `01-foundation/registry/models.json` and record known-good version for rollback.
- [ ] Review [alerting-runbook.md](./alerting-runbook.md) and [alerts/rules.yaml](../alerts/rules.yaml) before activation.

## Train Candidate

Use [scheduled_retraining.py](../production_patterns/scheduled_retraining.py) through Step 16 CLI so training, candidate registration, quality gate, optional activation, and machine-readable report stay in one scheduler-friendly path.

```bash
uv run production-scheduled-retrain \
  --config configs/foundation-recommender.yaml \
  --set-active \
  --require-quality-gate \
  --output-path 02-production-patterns/reports/scheduled-retraining.json
```

- [ ] Confirm command exits successfully.
- [ ] Confirm `02-production-patterns/reports/scheduled-retraining.json` has `status: "completed"`.
- [ ] Record `run_id`, `model_name`, and active target version.

## Quality Gate

- [ ] Confirm scheduled retraining report contains `quality_gate.status: "passed"`.
- [ ] If gate fails, stop release. Do not continue to serving verification.
- [ ] Inspect candidate metrics referenced by the retraining output before retrying.

## Activate Model

The scheduled retrain command above includes `--set-active`, so activation happens only after the quality gate passes.

- [ ] Confirm `01-foundation/registry/models.json` active entry points to intended version.
- [ ] Confirm old known-good version remains registered for rollback.
- [ ] If activation target is wrong, use rollback command in this checklist immediately.

## Serving Verification

- [ ] Start or restart serving API from project root if not already running.
- [ ] Check `GET /health` manually or through Step 15 monitor.
- [ ] Send a small prediction request to verify active model can serve traffic.
- [ ] Inspect `GET /metrics.json` for `prediction_error_count` and `prediction_latency_ms_last`.
- [ ] Inspect `GET /drift` for `drift_score`.

## Local Canary Decision

Use local canary decision after serving verification and drift check. Then run local traffic-splitting simulation before wider rollout.

```bash
uv run production-canary-decision \
  --deployment-demo 02-production-patterns/reports/local-deployment-demo.json \
  --drift-report 02-production-patterns/reports/local-deployment-drift.json \
  --approval 02-production-patterns/reports/approval-decision.json \
  --output-path 02-production-patterns/reports/local-canary-decision.json
```

- [ ] Confirm output has `status: "passed"` and `decision: "promote"`.
- [ ] If output has `decision: "rollback"`, use rollback plan before wider rollout.
- [ ] Record `canary_percent` and evidence paths in release notes.

```bash
uv run production-canary-router \
  --decision 02-production-patterns/reports/local-canary-decision.json \
  --stable-model-id foundation-config-v1 \
  --candidate-model-id local-candidate-v2 \
  --request-count 100 \
  --output-path 02-production-patterns/reports/local-canary-router.json
```

- [ ] Confirm candidate traffic matches `canary_percent`.
- [ ] Confirm rollback decisions route 100% traffic to stable model.

## Monitoring And Alerts

Use [monitoring_loop.py](../production_patterns/monitoring_loop.py) through Step 15 CLI after activation.

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

- [ ] Confirm output has `status: "healthy"`.
- [ ] Map any failed check to [alerts/rules.yaml](../alerts/rules.yaml): `high_error_count`, `high_latency`, `high_drift_score`, or `api_unhealthy`.
- [ ] Follow [alerting-runbook.md](./alerting-runbook.md) if monitor returns `status: "unhealthy"`.

## Rollback Plan

Use [rollback.py](../production_patterns/rollback.py) through Step 19 CLI if release verification fails, alerts fire, or active model is wrong.

```bash
uv run production-rollback-model \
  --registry-path 01-foundation/registry/models.json \
  --model-name movielens-popularity \
  --target-version foundation-config-v1 \
  --reason "release verification failed"
```

- [ ] Replace `foundation-config-v1` with recorded known-good version when needed.
- [ ] Confirm rollback output has `status: "rolled_back"`.
- [ ] Confirm `02-production-patterns/reports/rollback.json` records reason and target version.
- [ ] Rerun `uv run production-monitor ...` after rollback.

## Post-release Notes

- [ ] Save scheduled retraining report path.
- [ ] Save rollback report path if rollback happened.
- [ ] Record final active model version.
- [ ] Record monitor summary status and any alert checks.
- [ ] Capture follow-up work for CI/CD, deployment manifests, canary strategy, or real Alertmanager runtime.
