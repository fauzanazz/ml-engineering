# 02 Production Patterns

Purpose: expand from one ML app into common production ML application patterns.

## Progress Through Step 14

01 Foundation is closed at Step 10: train, artifact, config, experiment tracking, registry, local serving, metrics/logging, Dockerized API, and local monitoring stack.

Step 11 batch inference is treated as transition work: implementation still lives in `01-foundation/recommendation/batch.py`, but conceptually belongs here as a production pattern.

Step 12 added this production-patterns scaffold.
Step 13 added `production-retrain`.
Step 14 added quality gate checks before activation.

This folder now owns pattern-level docs and thin wrappers around foundation workflows.

## Pattern Docs

- [online serving](docs/online-serving.md)
- [batch inference](docs/batch-inference.md)
- [scheduled retraining](docs/retraining.md)
- [monitoring loop](docs/monitoring-loop.md)

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

## Target Learning Outcomes

- Compare multiple app shapes and service boundaries.
- Practice reusable training, evaluation, and serving patterns.
- Document tradeoffs for production readiness.
- Keep foundation code reusable while production patterns evolve separately.

## Status

Production pattern layer has started:

- batch inference wrapper exists
- retraining wrapper exists
- quality gate blocks unsafe activation

Airflow, scheduler, richer offline evaluation, rollback, and monitoring loop automation not added yet.
