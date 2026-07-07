# {{project_name}}

Deploy-agnostic ML model serving service template.

The service is intentionally model-free: `/predict` uses a deterministic dummy
implementation that returns a boolean. Replace `{{package_name}}/predict.py`
when you add a real model.

## Run locally

```bash
uv run serve
```

Smoke test:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"feature_a":1.0,"feature_b":-0.25}}'
```

## Test and build

```bash
uv run pytest
uv build
```

## Container

```bash
docker build -t {{package_name}} .
docker run --rm -p 8000:8000 {{package_name}}
```

This image only needs a container runtime. It is not tied to Kubernetes, ECS,
Heroku, a VPS, or any specific platform.

## Configuration

Configure the service with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind host inside the container/process. |
| `PORT` | `8000` | HTTP port. |
| `LOG_LEVEL` | `INFO` | Python/uvicorn log level. |
| `SERVICE_NAME` | `{{project_name}}` | OpenAPI service name. |
| `MODEL_NAME` | `{{package_name}}-dummy` | Model identifier in responses. |
| `MODEL_VERSION` | `0.1.0` | Model version in responses. |
| `PREDICTION_THRESHOLD` | `0.0` | Dummy boolean threshold for `sum(features)`. |

## Prediction contract

Request:

```json
{
  "features": {
    "feature_a": 1.0,
    "feature_b": -0.25
  }
}
```

Response:

```json
{
  "prediction": true,
  "model_name": "{{package_name}}-dummy",
  "model_version": "0.1.0"
}
```

Customize the contract by editing `PredictionRequest` and
`PredictionResponse` in `{{package_name}}/api.py`, then replacing
`predict()` in `{{package_name}}/predict.py`.
