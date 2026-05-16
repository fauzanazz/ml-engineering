---
title: Foundation YAML-Driven Training
type: feature-note
created: 2026-05-13
status: completed
categories: [recommendation, mlops, config, artifacts]
related:
  - ./step-1-foundation-scaffold-and-shared-architecture.md
  - ./step-3-local-experiment-tracking.md
  - ./step-4-local-model-registry.md
  - ../run-log.md
author: fauzan
---

# Foundation YAML-Driven Training

[Foundation recommender](../../src/ml_production_ecosystem/recommendation/) can train the popularity baseline from one YAML config and one command.

## Context

Step 1 made project structure and shared contracts. Step 2 moves foundation training closer to a real data-science workflow: a scientist edits `configs/foundation-recommender.yaml`, then runs `foundation-train-from-config` without changing Python code or passing many CLI flags.

This keeps [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) training simple while establishing a config-first habit before heavier platform pieces like [MLflow](https://mlflow.org/) or orchestration.

## How It Works

`foundation-train-from-config` loads `configs/foundation-recommender.yaml`, reads dataset paths, artifact directory, model type, `version`, and `min_rating`, then calls `train_popularity_recommender()` directly. Training logic was intentionally reused instead of rewritten.

Artifacts still use the old local format under `artifacts/foundation/recommendation/<version>/`:

- `model.json`
- `metadata.json`
- `metrics.json`

The config shape is intentionally small:

```yaml
pipeline:
  name: foundation-recommender
  version: foundation-config-v1

dataset:
  ratings_path: examples/samples/recommendation/ratings.csv
  movies_path: examples/samples/recommendation/movies.csv

model:
  type: popularity
  hyperparams:
    min_rating: 4.0

artifacts:
  artifact_dir: artifacts/foundation
```

## Key Files

| File | Role |
|------|------|
| `configs/foundation-recommender.yaml` | Example training config. |
| `src/ml_production_ecosystem/recommendation/train.py` | Config parser, CLI entrypoint, and reused popularity training call. |
| `tests/test_recommendation_workflow.py` | Test coverage for config-driven artifact creation. |
| `pyproject.toml` | Registers `foundation-train-from-config`. |

## Commands

```bash
uv run foundation-train-from-config --config configs/foundation-recommender.yaml
```

The command writes artifact files to:

```text
artifacts/foundation/recommendation/foundation-config-v1/
```

## Decisions & Trade-offs

We kept only `model.type: popularity` in scope because the milestone goal was config-driven foundation training, not algorithm expansion. Collaborative filtering and matrix factorization already exist in the CLI path, but Step 2 focuses on one stable baseline to reduce moving parts.

We added [PyYAML](https://pyyaml.org/) instead of hand-rolling YAML parsing. That is a small dependency but avoids fragile parsing as Step 3 and Step 4 add `experiments` and `registry` sections.

## Validation Results

Automated coverage verifies config-driven training writes the same artifact contract as direct training and preserves config-driven `version` and `min_rating`.

Latest full validation after Step 4:

```text
21 passed
```

## Known Limitations

The config path currently supports only popularity baseline training. Extra validation for malformed config keys is minimal and should be improved before treating this as a production config interface.

## Related

- [Step 3: Local Experiment Tracking](./step-3-local-experiment-tracking.md)
- [Step 4: Local Model Registry](./step-4-local-model-registry.md)
- [Foundation Movie Recommender Models and Local Infra](./step-2-foundation-recommender-models-and-local-infra.md)
