# Local Lifecycle Runbook

Purpose: verify local-first ML lifecycle without downloading MovieLens or using cloud services.

## Fast Smoke

```bash
./scripts/smoke-local-lifecycle.sh
```

Expected final line:

```text
local lifecycle smoke passed
```

The script runs `production-apply-local-platform`, then `production-lifecycle-demo` with `configs/local-lifecycle-demo.yaml` and checks:

- local filesystem resources are ready
- local secret injection references match the local platform plan
- platform plan validation passed
- local scheduler runtime can dry-run a planned job
- model contract manifest ready
- dataset manifest ready
- training completed
- offline validation passed
- graph artifacts written
- lifecycle status summary written

## Live Local Deployment Smoke

```bash
./scripts/smoke-local-deployment.sh
```

Expected final line:

```text
local deployment smoke passed
```

The script stays local-only and checks:

- lifecycle demo trains, approves, and activates sample model
- local FastAPI service starts from active registry pointer
- `/predict/v1` returns recommendations
- `production-demo-deployment` passes health, metrics, latency, and drift checks
- `production-detect-drift` writes passed drift report
- `production-canary-decision` promotes or blocks using demo, drift, and approval evidence
- `production-canary-router` simulates stable/candidate traffic split from canary decision
- `production-continual-decision` writes monitor/retrain decision
- `production-continual-summary` writes history trend summary
- `production-lifecycle-status` writes ready status summary

## Manual Steps

```bash
uv run production-apply-local-platform
uv run production-validate-local-secret-injections
uv run production-validate-local-kubernetes
uv run production-validate-local-scheduler
uv run production-run-local-scheduler --job-name lifecycle-status
uv run production-validate-platform-plan
uv run production-validate-model-contract --config configs/local-lifecycle-demo.yaml
uv run production-ingest-data --config configs/local-lifecycle-demo.yaml
uv run production-retrain --config configs/local-lifecycle-demo.yaml --require-quality-gate
uv run production-validate-offline --config configs/local-lifecycle-demo.yaml
uv run production-lifecycle-graph
uv run production-lifecycle-status
uv run production-goal-readiness
```

Optional after starting local API:

```bash
uv run foundation-serve-recommender --port 18080
uv run production-demo-deployment --base-url http://127.0.0.1:18080
uv run production-detect-drift --base-url http://127.0.0.1:18080
uv run production-canary-decision \
  --deployment-demo artifacts/reports/production-patterns/deployment-demo.json \
  --drift-report artifacts/reports/production-patterns/drift-report.json
uv run production-canary-router \
  --decision artifacts/reports/production-patterns/local-canary-decision.json \
  --stable-model-id foundation-config-v1 \
  --candidate-model-id local-candidate-v2
uv run production-continual-decision \
  --drift-report artifacts/reports/production-patterns/drift-report.json \
  --deployment-demo artifacts/reports/production-patterns/deployment-demo.json \
  --history-path artifacts/reports/production-patterns/continual-learning-history.jsonl
uv run production-continual-summary \
  --history-path artifacts/reports/production-patterns/continual-learning-history.jsonl
```

## Outputs

```text
artifacts/reports/production-patterns/local-platform-apply.json
artifacts/reports/production-patterns/local-secret-injections.json
artifacts/reports/production-patterns/local-kubernetes-validation.json
artifacts/reports/production-patterns/local-scheduler-validation.json
artifacts/reports/production-patterns/local-scheduler-run.json
artifacts/reports/production-patterns/local-lifecycle-demo.json
artifacts/reports/production-patterns/local-lifecycle-demo.mmd
artifacts/reports/production-patterns/local-lifecycle-demo.html
artifacts/reports/production-patterns/local-lifecycle-status.json
artifacts/reports/production-patterns/local-deployment-lifecycle.json
artifacts/reports/production-patterns/local-deployment-demo.json
artifacts/reports/production-patterns/local-deployment-drift.json
artifacts/reports/production-patterns/local-canary-decision.json
artifacts/reports/production-patterns/local-canary-router.json
artifacts/reports/production-patterns/local-deployment-status.json
artifacts/reports/production-patterns/platform-plan-validation.json
artifacts/reports/production-patterns/model-contract-manifest.json
artifacts/reports/production-patterns/dataset-manifest.json
artifacts/reports/production-patterns/offline-validation.json
artifacts/reports/production-patterns/approval-decision.json
artifacts/reports/production-patterns/deployment-demo.json
artifacts/reports/production-patterns/drift-report.json
artifacts/reports/production-patterns/continual-learning-decision.json
artifacts/reports/production-patterns/continual-learning-history.jsonl
artifacts/reports/production-patterns/continual-learning-summary.json
artifacts/reports/production-patterns/lifecycle-status.json
artifacts/reports/production-patterns/goal-readiness.json
```

These reports are generated artifacts. Do not commit them unless a future release policy says otherwise.
