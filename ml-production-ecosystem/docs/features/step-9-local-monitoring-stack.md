# Step 9: Local Monitoring Stack With Prometheus + Grafana

## Goal

Add local Prometheus + Grafana monitoring around the foundation FastAPI recommender `/metrics` endpoint.

## Local Services

| Service | URL | Purpose |
|---|---|---|
| Foundation API | `http://127.0.0.1:8000` | Serves predictions and Prometheus metrics. |
| Prometheus | `http://127.0.0.1:9090` | Scrapes `GET /metrics`. |
| Grafana | `http://127.0.0.1:3000` | Shows dashboard-ready observability. |

Prometheus scrape target from Docker:

```text
host.docker.internal:8000/metrics
```

## Run Flow

Run from `ml-production-ecosystem`.

### 1. Train/register model

```bash
uv run foundation-train-recommender \
  --ratings-path 01-foundation/data/raw/ml-25m/ratings.csv \
  --movies-path 01-foundation/data/raw/ml-25m/movies.csv \
  --artifact-dir 01-foundation/artifacts \
  --version api-v1
```

### 2. Set active model

```bash
uv run foundation-set-active-model \
  --registry-path 01-foundation/registry/models.json \
  --model-name movielens-popularity \
  --version api-v1
```

### 3. Start API

```bash
uv run foundation-serve-recommender --host 0.0.0.0 --port 8000
```

Confirm metrics locally:

```bash
curl http://127.0.0.1:8000/metrics
```

### 4. Start Prometheus and Grafana

```bash
docker compose up -d prometheus grafana
```

Open:

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

Grafana default local login remains upstream default:

```text
admin / admin
```

### 5. Make prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict/v1 \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 10, "top_k": 5}'
```

### 6. Inspect dashboard

Open Grafana at `http://127.0.0.1:3000`, then dashboard:

```text
Foundation / Foundation Recommender Observability
```

Panels cover:

- prediction requests total/rate
- prediction errors total/rate
- last latency
- latency average approximation
- model/version label visibility

## Provisioned Files

```text
monitoring/prometheus/prometheus.yml
monitoring/grafana/provisioning/datasources/prometheus.yml
monitoring/grafana/provisioning/dashboards/dashboards.yml
monitoring/grafana/provisioning/dashboards/foundation-recommender.json
```

## Notes

- This stack is local learning infra only.
- No alerting, Kubernetes, remote Prometheus, histogram buckets, or drift dashboard yet.
- API must run before Prometheus can scrape healthy targets.
