# Step 10: Dockerized Foundation API

## Goal

Package the foundation FastAPI recommender as a local Docker Compose service while keeping local train/register workflow.

## Flow

```text
local train/register/set active
  -> local artifacts/foundation + registry
  -> foundation-api container mounts runtime dirs
  -> Prometheus scrapes foundation-api:8000/metrics
  -> Grafana dashboard reads Prometheus
```

## Compose Service

`foundation-api` builds from local `Dockerfile` and runs:

```bash
foundation-serve-recommender --host 0.0.0.0 --port 8000
```

Mounted runtime dirs:

```text
artifacts/foundation -> /app/artifacts/foundation
registry  -> /app/registry
logs      -> /app/logs
```

Prediction logs persist in local `logs`.

## Run Flow

Run from `ml-production-ecosystem`.

### 1. Train/register model locally

```bash
uv run foundation-train-recommender \
  --ratings-path examples/samples/recommendation/ratings.csv \
  --movies-path examples/samples/recommendation/movies.csv \
  --artifact-dir artifacts/foundation \
  --version api-v1
```

### 2. Set active model locally

```bash
uv run foundation-set-active-model \
  --registry-path registry/models.json \
  --model-name movielens-popularity \
  --version api-v1
```

### 3. Start containerized API + monitoring

```bash
docker compose up -d foundation-api prometheus grafana
```

### 4. Confirm API and metrics

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

### 5. Make prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict/v1 \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 10, "top_k": 5}'
```

### 6. Inspect monitoring

Open:

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

Prometheus target:

```text
foundation-api:8000/metrics
```

Grafana dashboard:

```text
Foundation / Foundation Recommender Observability
```

## Host API Mode

Host mode remains available for development:

```bash
uv run foundation-serve-recommender --host 0.0.0.0 --port 8000
docker compose up -d prometheus grafana
```

For host mode, point Prometheus target at:

```text
host.docker.internal:8000
```

Default committed config uses Compose mode target `foundation-api:8000`.

## Not Included

- Kubernetes
- Dockerized training job
- remote registry/object storage
- CI image build
- auth/secrets
