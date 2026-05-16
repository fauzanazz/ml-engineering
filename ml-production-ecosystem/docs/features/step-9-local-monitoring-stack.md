# Step 9: Local Monitoring Stack With Prometheus + Grafana

## Goal

Add local Prometheus + Grafana monitoring around the foundation FastAPI recommender `/metrics` endpoint.

## Local Services

| Service | URL | Purpose |
|---|---|---|
| Foundation API | `http://127.0.0.1:8000` | Serves predictions and Prometheus metrics. |
| Prometheus | `http://127.0.0.1:9090` | Scrapes `GET /metrics`. |
| Grafana | `http://127.0.0.1:3000` | Shows dashboard-ready observability. |

Prometheus scrape target in full Compose mode:

```text
foundation-api:8000/metrics
```

Host API mode can still use `host.docker.internal:8000/metrics` if `monitoring/prometheus/prometheus.yml` is temporarily pointed at the host process.

## Run Flow

Run from `ml-production-ecosystem`.

### 1. Train/register model

```bash
uv run foundation-train-recommender \
  --ratings-path examples/samples/recommendation/ratings.csv \
  --movies-path examples/samples/recommendation/movies.csv \
  --artifact-dir artifacts/foundation \
  --version api-v1
```

### 2. Set active model

```bash
uv run foundation-set-active-model \
  --registry-path registry/models.json \
  --model-name movielens-popularity \
  --version api-v1
```

### 3A. Start host API mode

```bash
uv run foundation-serve-recommender --host 0.0.0.0 --port 8000
```

Then start monitoring only:

```bash
docker compose up -d prometheus grafana
```

Host API mode requires Prometheus target `host.docker.internal:8000`.

### 3B. Start compose API mode

```bash
docker compose up -d foundation-api prometheus grafana
```

Compose API mode uses Prometheus target `foundation-api:8000`.

Confirm metrics locally:

```bash
curl http://127.0.0.1:8000/metrics
```

### 4. Open Prometheus and Grafana

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
