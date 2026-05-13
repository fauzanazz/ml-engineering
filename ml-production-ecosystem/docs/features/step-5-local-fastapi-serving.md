---
title: Local FastAPI Serving
type: feature-note
created: 2026-05-13
status: completed
categories: [recommendation, mlops, serving, fastapi]
related:
  - ./step-2-yaml-driven-foundation-training.md
  - ./step-3-local-experiment-tracking.md
  - ./step-4-local-model-registry.md
  - ./step-2-foundation-recommender-models-and-local-infra.md
  - ../run-log.md
author: fauzan
---

# Local FastAPI Serving

[Foundation recommender](../../01-foundation/recommendation/) can serve the active registered model through local HTTP endpoints.

## Context

Step 4 introduced a local model registry so the project can answer which model version is active. Step 5 uses that active pointer for a small [FastAPI](https://fastapi.tiangolo.com/) serving layer, giving data scientists an HTTP prediction path without manually calling Python scripts.

This remains foundation scope. There is no auth, Docker image, async worker, Prometheus metrics, or load testing yet. The goal is to prove the path: train from config, register model, set active, then serve active model predictions.

## How It Works

`foundation-serve-recommender` starts `recommendation.api.create_app()` with a registry path. The API reads `01-foundation/registry/models.json`, resolves the active `movielens-popularity` model, and calls existing registry-backed prediction logic.

Endpoints:

| Method | Path | Role |
|--------|------|------|
| `GET` | `/health` | Returns service liveness. |
| `GET` | `/models/active` | Returns active model registry metadata. |
| `POST` | `/predict/v1` | Returns recommendations from the active artifact. |

Prediction request body:

```json
{
  "user_id": 10,
  "top_k": 5
}
```

Prediction response shape:

```json
{
  "model_name": "movielens-popularity",
  "version": "foundation-config-v1",
  "recommendations": []
}
```

If no active model exists, `/models/active` and `/predict/v1` return `404` with a clear message:

```json
{
  "detail": "active model not found: movielens-popularity"
}
```

## Key Files

| File | Role |
|------|------|
| `01-foundation/recommendation/api.py` | FastAPI app factory, endpoints, and serve CLI. |
| `01-foundation/recommendation/predict.py` | Registry-backed prediction helper reused by HTTP endpoint. |
| `01-foundation/recommendation/train.py` | Registry helpers used by serving to resolve active model metadata. |
| `tests/test_recommendation_api.py` | FastAPI TestClient coverage for health, active model, prediction, and missing-active behavior. |
| `pyproject.toml` | Adds `fastapi`, `uvicorn`, `httpx`, and `foundation-serve-recommender`. |

## Commands

Train from config:

```bash
uv run foundation-train-from-config --config configs/foundation-recommender.yaml
```

Set active model if config did not set it automatically:

```bash
uv run foundation-set-active-model \
  --registry-path 01-foundation/registry/models.json \
  --model-name movielens-popularity \
  --version foundation-config-v1
```

Serve locally:

```bash
uv run foundation-serve-recommender \
  --registry-path 01-foundation/registry/models.json \
  --host 127.0.0.1 \
  --port 8000
```

Call prediction endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict/v1 \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 10, "top_k": 5}'
```

## Decisions & Trade-offs

The app uses an app factory, `create_app(registry_path=...)`, so tests can inject a temporary registry. This avoids global mutable state and keeps serving deterministic.

The API resolves the active model on each request. That is simple and reflects registry changes immediately, but it reads local JSON per request. Step 6 observability can measure latency before deciding whether to cache.

`user_id` is optional because the popularity model does not need it. Future user-specific active models still receive `user_id` through the same request body.

## Validation Results

Tests cover:

- `/health` returns `{"status": "ok"}`
- `/models/active` returns active model metadata
- `/predict/v1` returns recommendations from active artifact
- missing active model returns `404`

Latest API test result:

```text
4 passed
```

## Known Limitations

No auth, rate limiting, Docker image, Prometheus metrics, drift monitoring, async worker, or load testing exists yet. Step 6 should add request count, latency, and error metrics before treating this as more than local serving.

## Related

- [Step 4: Local Model Registry](./step-4-local-model-registry.md)
- [Step 3: Local Experiment Tracking](./step-3-local-experiment-tracking.md)
- [Step 2: YAML-Driven Foundation Training](./step-2-yaml-driven-foundation-training.md)
