---
title: Local Experiment Tracking
type: feature-note
created: 2026-05-13
status: completed
categories: [recommendation, mlops, experiments, reproducibility]
related:
  - ./step-2-yaml-driven-foundation-training.md
  - ./step-4-local-model-registry.md
  - ./step-2-foundation-recommender-models-and-local-infra.md
  - ../run-log.md
author: fauzan
---

# Local Experiment Tracking

[Foundation recommender](../../01-foundation/recommendation/) now records every config-driven training run as a local experiment folder.

## Context

Step 2 made training runnable from one YAML config. Step 3 captures the result of that run so data scientists can inspect training history without manually opening artifact folders.

This is intentionally not [MLflow](https://mlflow.org/). The goal is to introduce experiment-tracking concepts with plain files first: config snapshot, params, metrics, artifact references, timestamp, and run status.

## How It Works

`train_recommender_from_config()` reads the optional `experiments` section:

```yaml
experiments:
  tracking_dir: 01-foundation/experiments/runs
  run_id: foundation-config-v1
```

When `tracking_dir` is present, training writes one run folder:

```text
01-foundation/experiments/runs/<run_id>/
├── artifact.json
├── config.yaml
├── metrics.json
├── params.json
└── run.json
```

`config.yaml` copies the original config so the run is reproducible. `params.json` captures selected model params. `metrics.json` copies the training metrics from the model artifact. `artifact.json` records artifact and metrics URIs. `run.json` records run identity and status.

Example `run.json` shape:

```json
{
  "run_id": "foundation-config-v1",
  "model_name": "movielens-popularity",
  "version": "foundation-config-v1",
  "artifact_uri": "01-foundation/artifacts/recommendation/foundation-config-v1",
  "metrics_uri": "01-foundation/artifacts/recommendation/foundation-config-v1/metrics.json",
  "status": "completed",
  "created_at": "2026-05-13T..."
}
```

## Key Files

| File | Role |
|------|------|
| `01-foundation/recommendation/train.py` | Writes local experiment run records and lists run metadata. |
| `configs/foundation-recommender.yaml` | Adds `experiments.tracking_dir` and optional `experiments.run_id`. |
| `01-foundation/experiments/runs/` | Ignored local run-history storage. |
| `tests/test_recommendation_workflow.py` | Verifies run record creation and listing. |
| `.gitignore` | Ignores local experiment outputs. |

## Commands

Train and record a run:

```bash
uv run foundation-train-from-config --config configs/foundation-recommender.yaml
```

List recorded runs:

```bash
uv run foundation-list-runs --tracking-dir 01-foundation/experiments/runs
```

## Decisions & Trade-offs

Run records are copied, not symlinked. This makes each run folder self-describing and easier to inspect, at the cost of duplicating `metrics.json`.

Run status currently records only `completed`. Failure tracking is out of scope for Step 3 but should be added before this becomes operational tooling.

We keep experiment outputs ignored by git because they are local execution history, not source-of-truth project config.

## Validation Results

Tests cover:

- config-driven training still creates model artifact
- experiment folder is created from config
- original config is copied into `config.yaml`
- params, metrics, artifact metadata, and run metadata are written
- `list_experiment_runs()` returns run records

Latest full validation after Step 4:

```text
21 passed
```

## Known Limitations

There is no run comparison UI, no failure-state record, and no artifact garbage collection. This is a local file foundation that can later map cleanly to MLflow-style experiment tracking.

## Related

- [Step 2: YAML-Driven Foundation Training](./step-2-yaml-driven-foundation-training.md)
- [Step 4: Local Model Registry](./step-4-local-model-registry.md)
- [Foundation Movie Recommender Models and Local Infra](./step-2-foundation-recommender-models-and-local-infra.md)
