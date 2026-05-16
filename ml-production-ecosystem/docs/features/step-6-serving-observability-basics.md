---
title: Serving Observability Basics
type: feature-note
created: 2026-05-13
status: completed
categories: [recommendation, mlops, serving, observability]
related:
  - ./step-5-local-fastapi-serving.md
  - ./step-7-prediction-logging-and-basic-drift-signal.md
  - ./step-4-local-model-registry.md
  - ./step-2-foundation-recommender-models-and-local-infra.md
  - ../run-log.md
author: fauzan
---

# Serving Observability Basics

[Foundation recommender API](../../src/ml_production_ecosystem/recommendation/api.py) now exposes in-memory serving metrics for local prediction monitoring.

## Context

Step 5 made the active registered model available over HTTP. Step 6 makes that endpoint less of a black box by tracking prediction request volume, errors, latency, and last-used model tags.

This is intentionally not [Prometheus](https://prometheus.io/), [Grafana](https://grafana.com/), or [OpenTelemetry](https://opentelemetry.io/). The goal is a local JSON metrics baseline that proves the serving path can answer basic operational questions before adding production observability tools.

## How It Works

`recommendation.api.ServingMetrics` is an in-memory collector owned by each `create_app()` instance. `/predict/v1` records timing with `perf_counter()`, increments request count for every prediction attempt, increments error count for missing active model or prediction failures, and records the last successful model name/version.

The prediction response now includes `request_id` so a caller can correlate one HTTP response with logs or future prediction audit records:

```json
{
  "request_id": "...",
  "model_name": "movielens-popularity",
  "version": "api-v1",
  "recommendations": []
}
```

`GET /metrics` returns JSON:

```json
{
  "prediction_request_count": 3,
  "prediction_error_count": 1,
  "prediction_latency_ms_avg": 2.4,
  "prediction_latency_ms_last": 1.9,
  "last_model_name": "movielens-popularity",
  "last_model_version": "api-v1"
}
```

## Key Files

| File | Role |
|------|------|
| `src/ml_production_ecosystem/recommendation/api.py` | `ServingMetrics`, `/metrics`, request timing, request IDs, error accounting. |
| `tests/test_recommendation_api.py` | Coverage for metrics before/after success, request ID, and missing-active error count. |
| `docs/features/step-5-local-fastapi-serving.md` | Serving endpoint docs updated with metrics endpoint. |

## Endpoints

| Method | Path | Role |
|--------|------|------|
| `GET` | `/health` | Service liveness. |
| `GET` | `/models/active` | Active registry model metadata. |
| `GET` | `/metrics` | In-memory prediction metrics as JSON. |
| `POST` | `/predict/v1` | Prediction plus request ID. |

## Commands

Start local serving:

```bash
uv run foundation-serve-recommender \
  --registry-path registry/models.json \
  --host 127.0.0.1 \
  --port 8000
```

Read metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

## Decisions & Trade-offs

Metrics are process-local and reset on server restart. That is acceptable for foundation learning but not durable enough for production operations.

Latency averages include both successful and failed prediction attempts. This keeps request accounting simple and makes failure latency visible, but it does not yet separate success latency from error latency.

`last_model_name` and `last_model_version` update only when a model was resolved. Missing-active errors increment error count but leave model tags unchanged.

## Validation Results

Tests cover:

- initial `/metrics` state
- successful `/predict/v1` increments request count
- response includes `request_id`
- latency metrics are recorded after prediction
- missing active model increments error count

Latest API test result:

```text
5 passed
```

## Known Limitations

Metrics are not Prometheus-formatted, not persistent, and not safe for multi-process aggregation. There is no request log or drift signal yet. Step 7 should add prediction logging and a basic drift indicator.

## Related

- [Step 5: Local FastAPI Serving](./step-5-local-fastapi-serving.md)
- [Step 4: Local Model Registry](./step-4-local-model-registry.md)
- [Step 3: Local Experiment Tracking](./step-3-local-experiment-tracking.md)
