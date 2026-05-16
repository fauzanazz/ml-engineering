---
title: Live Foundation API Smoke Test
type: feature-note
created: 2026-05-15
status: completed
categories: [ml-production, serving, smoke-test]
related:
  - ./production-compose.md
  - ./release-checklist.md
  - ../deploy/deployment-manifest.yaml
  - ./alerting-runbook.md
---

# Live Foundation API Smoke Test

`./scripts/smoke-test-foundation-api.sh` verifies a running `foundation-api` before release. It checks the endpoints named by `deployment-manifest.yaml` and used by [release-checklist.md](./release-checklist.md), then leaves longer threshold checks to `production-monitor`.

## 1. Start production compose

Run from `ml-production-ecosystem`:

```bash
docker compose -f docker-compose.production.yaml up --build foundation-api
```

See [production-compose.md](./production-compose.md) for compose details, mounted registry/artifacts paths, and stop command.

## 2. Run live smoke test

In another shell, run:

```bash
./scripts/smoke-test-foundation-api.sh http://127.0.0.1:8000
```

If no URL is passed, the script defaults to `http://127.0.0.1:8000`.

The script fails fast when any HTTP request fails, any endpoint returns non-JSON where JSON is required, or the prediction response has no recommendations.

Checks:

- `GET /health` returns `status: ok`
- `GET /metrics.json` returns JSON with prediction metric fields
- `GET /drift` returns JSON with drift fields
- `POST /predict/v1` with a small `top_k` payload returns global baseline recommendations

## 3. Run production monitor

After smoke passes, run the production monitor thresholds used by production compose verification:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

## Release usage

Use this flow between serving startup and release signoff:

1. Start [production compose](./production-compose.md).
2. Run `./scripts/smoke-test-foundation-api.sh http://127.0.0.1:8000`.
3. Run `uv run production-monitor` with release thresholds.
4. Continue [release-checklist.md](./release-checklist.md) only if smoke and monitor both pass.

This script does not start Docker automatically, run load tests, configure auth, send canary traffic, or perform long-running soak tests.
