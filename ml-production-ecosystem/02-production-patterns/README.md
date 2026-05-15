# 02 Production Patterns

Purpose: expand from one ML app into common production ML application patterns.

## Progress Through Step 27

01 Foundation is closed at Step 10: train, artifact, config, experiment tracking, registry, local serving, metrics/logging, Dockerized API, and local monitoring stack.

Step 11 batch inference is treated as transition work: implementation still lives in `01-foundation/recommendation/batch.py`, but conceptually belongs here as a production pattern.

Step 12 added this production-patterns scaffold.
Step 13 added `production-retrain`.
Step 14 added quality gate checks before activation.
Step 15 added `production-monitor` for health, metrics, and drift checks.
Step 16 added `production-scheduled-retrain` with machine-readable reports.
Step 17 added an import-safe Airflow retraining DAG skeleton.
Step 18 added alert rules and an alerting runbook.
Step 19 added `production-rollback-model`.
Step 20 added the production model release checklist.
Step 21 added the deployment manifest and companion docs.
Step 22 added focused local CI validation.
Step 23 added GitHub Actions remote CI skeleton.
Step 24 added production-like Docker Compose runtime.
Step 25 added live API smoke testing.
Step 26 added `production-release-summary`.
Step 27 added the production patterns scope review and closure checklist.

This folder now owns pattern-level docs, operational runbooks, and thin wrappers around foundation workflows.

## Pattern Docs

- [online serving](docs/online-serving.md)
- [batch inference](docs/batch-inference.md)
- [scheduled retraining](docs/retraining.md)
- [monitoring loop](docs/monitoring-loop.md)
- [alerting runbook](docs/alerting-runbook.md)
- [release checklist](docs/release-checklist.md)
- [deployment manifest](docs/deployment-manifest.md)
- [local CI](docs/local-ci.md)
- [production compose](docs/production-compose.md)
- [live smoke test](docs/live-smoke-test.md)
- [incident simulation](docs/incident-simulation.md)
- [AWS serving decision](docs/aws-serving-decision.md)
- [security controls](docs/security-controls.md)
- [scope review](docs/scope-review.md)

## Current Wrappers

Batch inference:

```bash
uv run production-batch-recommend \
  --registry-path 01-foundation/registry/models.json \
  --input-path 01-foundation/data/batch/input.jsonl \
  --output-path 01-foundation/logs/batch-output.jsonl
```

Retraining:

```bash
uv run production-retrain --config configs/foundation-recommender.yaml --set-active --require-quality-gate
```

Monitoring:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

Scheduled retraining:

```bash
uv run production-scheduled-retrain \
  --config configs/foundation-recommender.yaml \
  --set-active \
  --require-quality-gate \
  --output-path 02-production-patterns/reports/scheduled-retraining.json
```

Rollback:

```bash
uv run production-rollback-model \
  --registry-path 01-foundation/registry/models.json \
  --model-name movielens-popularity \
  --target-version foundation-config-v1 \
  --reason "release verification failed"
```

Release summary:

```bash
uv run production-release-summary \
  --output-path 02-production-patterns/reports/release-summary.json
```

## Target Learning Outcomes

- Compare multiple app shapes and service boundaries.
- Practice reusable training, evaluation, and serving patterns.
- Document tradeoffs for production readiness.
- Keep foundation code reusable while production patterns evolve separately.

## Status

Production pattern layer is current through Step 27:

- batch inference wrapper exists
- retraining wrapper exists
- quality gate blocks unsafe activation
- monitoring loop checks serving health, metrics, and drift
- scheduled retraining writes machine-readable reports
- Airflow DAG skeleton imports safely without requiring scheduler runtime
- alert rules and runbooks document triage
- rollback command restores known-good active model state
- release checklist and release summary capture operational evidence
- deployment manifest records service endpoints and rollback command
- local CI script validates focused production-pattern tests
- GitHub Actions workflow runs the same validation on push/PR
- production compose and smoke test verify local serving runtime
- release summary captures release evidence
- incident simulation, security controls, and AWS serving decision docs capture extra operational hardening

MLflow stages, Kubernetes, real cloud deployment, remote scheduler runtime, managed secrets, canary deployment, load testing, and `03-million-scale` implementation are not added yet.
