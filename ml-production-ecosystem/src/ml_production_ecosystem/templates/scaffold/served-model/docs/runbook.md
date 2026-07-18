# Serving Runbook

## Verify and train

```bash
uv run pytest
uv run python -m {{package_name}}.train
```

Training writes `artifacts/reports/training-summary.json` and
`artifacts/reports/metrics.json`. It does not change the live model: the
generated service deliberately keeps deterministic dummy inference until
`{{package_name}}/predict.py` is replaced.

## Start

```bash
uv run serve
```

Container:

```bash
docker build -t {{package_name}} .
docker run --rm -p 8000:8000 {{package_name}}
```

## Smoke test

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"feature_a":1.0,"feature_b":-0.25}}'
```

## Build

```bash
uv build
docker build -t {{package_name}} .
```

## Operate

- Health check: `GET /health`
- Inference: `POST /predict`
- Logs: stdout/stderr, suitable for any container platform log collector.
- Config: `HOST`, `PORT`, `LOG_LEVEL`, `SERVICE_NAME`, `MODEL_NAME`, `MODEL_VERSION`, `PREDICTION_THRESHOLD`

## Customize

Replace `{{package_name}}/predict.py` with real model loading/inference. If
your input or output shape changes, update `PredictionRequest`,
`PredictionResponse`, and `configs/project.yaml` together.
