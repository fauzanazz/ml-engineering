---
title: Step 8 Prometheus-Style Metrics Endpoint
type: feature-note
created: 2026-05-14
status: completed
categories: [foundation, observability, prometheus, fastapi]
related:
  - ./step-6-serving-observability-basics.md
  - ./step-9-local-monitoring-stack.md
  - ../../docs/domains/foundation/README.md
---

Step 8 upgraded the foundation recommender metrics API from local JSON-only output to Prometheus-compatible text.

## Context

[Step 6 serving observability](./step-6-serving-observability-basics.md) introduced local API metrics as a JSON snapshot. That was useful for early debugging, but not enough for a future monitoring stack. Step 8 changed the endpoint shape so [Prometheus](https://prometheus.io/) can scrape the local [FastAPI](https://fastapi.tiangolo.com/) app while keeping the old JSON view available for compatibility.

This step intentionally stayed lightweight. It did not add a real Prometheus server, Grafana dashboard, histogram buckets, multiprocessing support, OpenTelemetry, or alerting. Those were deferred to [Step 9 local monitoring stack](./step-9-local-monitoring-stack.md).

## API Endpoints

| Endpoint | Response | Purpose |
|---|---|---|
| `GET /metrics` | `text/plain` Prometheus-style metrics | Scrape-ready production observability shape. |
| `GET /metrics.json` | JSON snapshot | Backward-compatible local metrics view from Step 6. |

## Prometheus Metrics

The endpoint exposes prediction counters and latency summary-like values:

```text
foundation_prediction_requests_total{model_name="movielens-popularity",model_version="api-v1"} 3
foundation_prediction_errors_total{model_name="movielens-popularity",model_version="api-v1"} 1
foundation_prediction_latency_ms_sum{model_name="movielens-popularity",model_version="api-v1"} 12.4
foundation_prediction_latency_ms_count{model_name="movielens-popularity",model_version="api-v1"} 3
foundation_prediction_latency_ms_last{model_name="movielens-popularity",model_version="api-v1"} 1.9
```

Labels:

| Label | Meaning |
|---|---|
| `model_name` | Active model family, for example `movielens-popularity`. |
| `model_version` | Active registered model version, for example `api-v1`. |

When no prediction has resolved a model yet, metrics use `unknown` label values.

## Behavior

Successful predictions increment `foundation_prediction_requests_total` and update latency metrics. Failed predictions increment both `foundation_prediction_requests_total` and `foundation_prediction_errors_total`. The JSON snapshot remains available through `/metrics.json` so older local checks do not break.

## Tests

Coverage lives in [`tests/test_recommendation_api.py`](../../tests/test_recommendation_api.py). Tests verify:

- `/metrics` returns `text/plain`
- `/metrics` contains Prometheus metric names and labels
- `/metrics.json` keeps the Step 6 JSON snapshot shape
- successful predictions increment request metrics
- failed predictions increment error metrics
- existing API behavior remains green

## Decision Notes

Prometheus text format was added directly instead of adding the Prometheus Python client. This kept foundation simple and avoided histogram/multiprocess concerns before a real monitoring stack existed. Step 9 then consumed this endpoint without requiring another API change.
