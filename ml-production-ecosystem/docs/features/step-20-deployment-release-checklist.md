# Step 20: Deployment Release Checklist

## Goal

Add a release checklist for safe manual model deployment from config to active serving state.

## User Story

Sebagai ML engineer, gw punya checklist rilis model: train, quality gate, activate, monitor, alert check, rollback plan. Jadi deploy tidak asal jalan command.

## Checklist Sections

The checklist lives at `docs/domains/production-patterns/release-checklist.md` and covers:

- Pre-release.
- Train Candidate.
- Quality Gate.
- Activate Model.
- Serving Verification.
- Monitoring And Alerts.
- Rollback Plan.
- Post-release Notes.

The workflow references [Step 16 scheduled retraining](./step-16-scheduled-retraining-entrypoint.md), [Step 15 monitoring](./step-15-monitoring-loop-entrypoint.md), [Step 18 alerts](./step-18-alerting-rules-and-runbook.md), and [Step 19 rollback](./step-19-model-rollback-entrypoint.md).

## Commands

Train candidate, quality gate, and activate:

```bash
uv run production-scheduled-retrain \
  --config configs/foundation-recommender.yaml \
  --set-active \
  --require-quality-gate \
  --output-path artifacts/reports/production-patterns/scheduled-retraining.json
```

Monitor after activation:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

Rollback if verification fails:

```bash
uv run production-rollback-model \
  --registry-path registry/models.json \
  --model-name movielens-popularity \
  --target-version foundation-config-v1 \
  --reason "release verification failed"
```

## Key Files

- `docs/domains/production-patterns/release-checklist.md`
- `src/ml_production_ecosystem/production_patterns/scheduled_retraining.py`
- `src/ml_production_ecosystem/production_patterns/monitoring_loop.py`
- `src/ml_production_ecosystem/production_patterns/rollback.py`
- `configs/production-patterns/alerts/rules.yaml`
- `tests/test_release_checklist.py`

## Pattern

```text
pre-release review
  -> scheduled retrain
  -> quality gate
  -> activate model
  -> serving verification
  -> monitor + alert review
  -> rollback if needed
  -> post-release notes
```

## Out Of Scope

- Automated release approval.
- CI/CD pipeline.
- Kubernetes deployment.
- Canary traffic split.
- Blue/green deploy.
- Human approval system.

## Acceptance Criteria

- Release checklist doc exists.
- Checklist references Step 16 scheduled retrain, Step 15 monitor, Step 18 alerts, Step 19 rollback.
- Commands are copy-paste runnable from project root.
- Tests assert checklist has required sections and commands.
- Existing tests stay green.

## Definition Of Done

`production-patterns domain` has manual production release workflow. Project covers train → gate → activate → serve → monitor → alert → rollback.

## Next Step

Step 21 can add deployment manifest or lightweight CI validation. This project added deployment manifest next.
