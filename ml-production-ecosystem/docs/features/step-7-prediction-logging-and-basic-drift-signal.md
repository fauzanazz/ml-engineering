---
title: Prediction Logging and Basic Drift Signal
type: feature-note
created: 2026-05-13
status: completed
categories: [recommendation, mlops, serving, observability, drift]
related:
  - ./step-6-serving-observability-basics.md
  - ./step-5-local-fastapi-serving.md
  - ./step-4-local-model-registry.md
  - ./step-2-foundation-recommender-models-and-local-infra.md
  - ../run-log.md
author: fauzan
---

# Prediction Logging and Basic Drift Signal

[Foundation recommender API](../../src/ml_production_ecosystem/recommendation/api.py) now writes local prediction audit logs and exposes a naive output drift signal.

## Context

Step 6 added in-memory serving metrics for request count, latency, and errors. Step 7 adds an audit trail so serving behavior can be inspected after requests complete, and introduces a simple drift score based on recommendation movie IDs.

This is not statistical drift detection. It is a foundation signal that compares recent recommendation outputs against the active artifact's baseline top recommendations.

## How It Works

`foundation-serve-recommender` accepts `--prediction-log-path`, defaulting to:

```text
logs/predictions.jsonl
```

Every `/predict/v1` attempt appends one JSONL row. Successful rows include the request ID returned to the caller, request inputs, model identity, recommended movie IDs, status, latency, and no error. Error rows still record request ID, request inputs, status, latency, and error message.

Successful log row shape:

```json
{
  "request_id": "...",
  "requested_at": "2026-05-13T...",
  "user_id": 10,
  "top_k": 5,
  "model_name": "movielens-popularity",
  "model_version": "api-v1",
  "recommendation_movie_ids": [1, 3],
  "status": "success",
  "latency_ms": 1.9,
  "error": null
}
```

Error log row shape:

```json
{
  "request_id": "...",
  "requested_at": "2026-05-13T...",
  "user_id": null,
  "top_k": 5,
  "model_name": null,
  "model_version": null,
  "recommendation_movie_ids": [],
  "status": "error",
  "latency_ms": 1.2,
  "error": "active model not found: movielens-popularity"
}
```

`GET /drift` reads recent successful prediction logs, compares their `recommendation_movie_ids` to the active artifact's top recommendations, and returns a score from `0.0` to `1.0`. A score of `0.0` means recent output IDs all appear in the active baseline set; higher scores mean more recent IDs fall outside that baseline.

Drift response shape:

```json
{
  "model_name": "movielens-popularity",
  "version": "api-v1",
  "sample_size": 10,
  "baseline_size": 10,
  "drift_score": 0.2
}
```

## Key Files

| File | Role |
|------|------|
| `src/ml_production_ecosystem/recommendation/api.py` | JSONL prediction logging, `--prediction-log-path`, and `/drift`. |
| `tests/test_recommendation_api.py` | Coverage for success logs, error logs, drift response, metrics compatibility. |
| `.gitignore` | Ignores local `logs/` outputs. |

## Commands

Serve with default prediction log path:

```bash
uv run foundation-serve-recommender \
  --registry-path registry/models.json \
  --host 127.0.0.1 \
  --port 8000
```

Serve with explicit prediction log path:

```bash
uv run foundation-serve-recommender \
  --registry-path registry/models.json \
  --prediction-log-path logs/predictions.jsonl \
  --host 127.0.0.1 \
  --port 8000
```

Read drift:

```bash
curl http://127.0.0.1:8000/drift
```

## Decisions & Trade-offs

Logs are local JSONL files because they are inspectable, append-friendly, and easy to test. There is no database audit table yet.

The drift score uses output movie IDs instead of feature distributions. That is intentionally naive, but it creates the first monitoring seam for prediction behavior changes.

The drift baseline uses the active artifact's top recommendations, so changing the active model changes the baseline immediately.

## Validation Results

Tests cover:

- success prediction appends one JSONL row
- error prediction appends one JSONL row
- log row `request_id` matches `/predict/v1` response
- `/drift` returns model name, version, sample size, baseline size, and drift score
- existing `/metrics` behavior still works

Latest API test result:

```text
7 passed
```

## Known Limitations

JSONL logging is synchronous and local. Drift is not statistically meaningful, does not compare input feature distributions, and does not alert. Step 8 should move toward stronger monitoring or durable serving operations.

## Related

- [Step 6: Serving Observability Basics](./step-6-serving-observability-basics.md)
- [Step 5: Local FastAPI Serving](./step-5-local-fastapi-serving.md)
- [Step 4: Local Model Registry](./step-4-local-model-registry.md)
