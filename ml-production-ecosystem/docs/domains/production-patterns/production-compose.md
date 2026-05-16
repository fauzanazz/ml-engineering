---
title: Production Compose Profile
type: feature-note
created: 2026-05-15
status: completed
categories: [ml-production, serving, compose]
related:
  - ../deploy/deployment-manifest.yaml
  - ./deployment-manifest.md
  - ./release-checklist.md
---

# Production Compose Profile

`docker-compose.production.yaml` starts the `foundation-api` service using deployment manifest conventions: image `ml-production-ecosystem-foundation-api`, command `uv run foundation-serve-recommender --host 0.0.0.0 --port 8000 --prediction-log-path logs/production-compose-predictions.jsonl`, port `8000:8000`, and health endpoint `/health`.

The compose file mounts local `artifacts/foundation`, `registry`, and `logs` paths into `/app` so the running API keeps using local model artifacts and registry state. Production compose writes predictions to `logs/production-compose-predictions.jsonl` to avoid drift checks reading unrelated local test traffic.

## Start

Run from `ml-production-ecosystem`:

```bash
docker compose -f docker-compose.production.yaml up --build foundation-api
```

## Verify

Health, metrics, and drift endpoints stay aligned with `deployment-manifest.yaml` and `release-checklist.md`:

- `/health`
- `/metrics`
- `/metrics.json`
- `/drift`

Run the production monitor against the local service:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

## Stop

In another shell, run:

```bash
docker compose -f docker-compose.production.yaml down
```

## Limits

This is a local production-like runtime profile only. It does not configure Kubernetes, TLS, autoscaling, multi-replica serving, a load balancer, image push, or cloud deployment.
