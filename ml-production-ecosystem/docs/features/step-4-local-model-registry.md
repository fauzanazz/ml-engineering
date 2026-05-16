---
title: Local Model Registry
type: feature-note
created: 2026-05-13
status: completed
categories: [recommendation, mlops, registry, artifacts]
related:
  - ./step-2-yaml-driven-foundation-training.md
  - ./step-3-local-experiment-tracking.md
  - ./step-5-local-fastapi-serving.md
  - ./step-2-foundation-recommender-models-and-local-infra.md
  - ../run-log.md
author: fauzan
---

# Local Model Registry

[Foundation recommender](../../src/ml_production_ecosystem/recommendation/) now has a human-readable local registry for registered and active model versions.

## Context

Step 1 created project boundaries. Step 2 made training config-driven. Step 3 recorded experiment runs. Step 4 answers the next operational question: after many artifacts exist, which model versions are known, and which version is active?

This registry is deliberately lightweight. It is a JSON file, not [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html), because the foundation milestone needs visible mechanics before introducing platform weight.

## How It Works

`train_recommender_from_config()` reads the optional `registry` section:

```yaml
registry:
  path: registry/models.json
  stage: candidate
  set_active: false
```

When `registry.path` is present, the trained artifact is registered into `models.json` with model name, version, artifact URI, metrics URI, stage, and creation timestamp.

Registry shape:

```json
{
  "models": [
    {
      "model_name": "movielens-popularity",
      "version": "foundation-config-v1",
      "artifact_uri": "artifacts/foundation/recommendation/foundation-config-v1",
      "metrics_uri": "artifacts/foundation/recommendation/foundation-config-v1/metrics.json",
      "stage": "candidate",
      "created_at": "2026-05-13T..."
    }
  ],
  "active": {
    "movielens-popularity": "foundation-config-v1"
  }
}
```

The active pointer lets future prediction and serving paths resolve a model by registry state instead of hardcoding an artifact folder.

## Key Files

| File | Role |
|------|------|
| `src/ml_production_ecosystem/recommendation/train.py` | Registry helpers, config integration, and registry CLIs. |
| `src/ml_production_ecosystem/recommendation/predict.py` | Active-model prediction helper and `--registry-path` CLI support. |
| `configs/foundation-recommender.yaml` | Adds `registry.path`, `registry.stage`, and `registry.set_active`. |
| `registry/models.json` | Ignored local registry state. |
| `tests/test_recommendation_workflow.py` | Covers register/list/get/set-active and active registry prediction. |

## API Surface

Registry helpers:

```python
register_model_version(
    registry_path: Path,
    model_name: str,
    version: str,
    artifact_uri: str,
    metrics_uri: str,
    stage: str = "candidate",
    set_active: bool = False,
) -> dict[str, object]
```

```python
list_model_versions(registry_path: Path, model_name: str | None = None) -> list[dict[str, object]]
get_model_version(registry_path: Path, model_name: str, version: str) -> dict[str, object] | None
get_active_model(registry_path: Path, model_name: str) -> dict[str, object] | None
set_active_model(registry_path: Path, model_name: str, version: str) -> dict[str, object]
```

Prediction from active registry model:

```python
recommend_top_k_from_registry(
    registry_path: Path,
    model_name: str = "movielens-popularity",
    top_k: int = 10,
    user_id: int | None = None,
) -> list[dict[str, object]]
```

## Commands

Train and optionally register:

```bash
uv run foundation-train-from-config --config configs/foundation-recommender.yaml
```

List registered model versions:

```bash
uv run foundation-list-models --registry-path registry/models.json
```

Set active model:

```bash
uv run foundation-set-active-model \
  --registry-path registry/models.json \
  --model-name movielens-popularity \
  --version foundation-config-v1
```

Predict from active registry model:

```bash
uv run foundation-recommend \
  --registry-path registry/models.json \
  --top-k 5
```

## Decisions & Trade-offs

The registry stores local paths as strings rather than abstract storage URIs. This matches the current local foundation scope and keeps the JSON easy to inspect.

Registering the same `model_name` and `version` replaces the prior entry. This prevents duplicate rows for reruns with the same version, but it also means rerun history belongs in [Local Experiment Tracking](./step-3-local-experiment-tracking.md), not the registry.

`stage` is metadata only. There is no approval workflow yet. `active` is the operational pointer that future serving should use.

## Validation Results

Tests cover:

- registering a model version
- listing registered versions
- getting a specific version
- setting and reading active model
- config-driven training writing registry entry
- prediction resolving active artifact from registry

Latest full validation:

```text
21 passed
```

## Known Limitations

There is no remote registry, locking, concurrent write protection, audit history, or approval workflow. This is enough for local model selection and prepares Step 5 serving to load the active model.

## Related

- [Step 2: YAML-Driven Foundation Training](./step-2-yaml-driven-foundation-training.md)
- [Step 3: Local Experiment Tracking](./step-3-local-experiment-tracking.md)
- [Foundation Movie Recommender Models and Local Infra](./step-2-foundation-recommender-models-and-local-infra.md)
