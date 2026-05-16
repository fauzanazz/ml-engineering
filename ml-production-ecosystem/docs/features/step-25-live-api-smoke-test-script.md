# Step 25: Live API Smoke Test Script

## Goal

Add a manual smoke test script that verifies a running `foundation-api` responds on health, metrics JSON, drift, and prediction path.

## User Story

Sebagai ML engineer, setelah start compose production, gw bisa run satu command buat cek API hidup dan endpoint penting jalan sebelum release dianggap aman.

## Command

Start production compose first:

```bash
docker compose -f docker-compose.production.yaml up --build foundation-api
```

Run smoke test in another shell:

```bash
./scripts/smoke-test-foundation-api.sh http://127.0.0.1:8000
```

If no URL is passed, the script defaults to:

```text
http://127.0.0.1:8000
```

Run monitor after smoke passes:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

## Checks

The smoke test verifies:

- `GET /health` returns `status: ok`.
- `GET /metrics.json` returns JSON.
- `GET /drift` returns JSON.
- one small prediction/recommendation request succeeds.

The script fails fast on HTTP/API failure so release verification stops early.

## Key Files

- `scripts/smoke-test-foundation-api.sh`
- `docs/domains/production-patterns/live-smoke-test.md`
- `docs/domains/production-patterns/production-compose.md`
- `docs/domains/production-patterns/release-checklist.md`
- `configs/production-patterns/deploy/deployment-manifest.yaml`

## Pattern

```text
production compose running
  -> smoke-test-foundation-api.sh
  -> health/metrics/drift/predict checks
  -> production-monitor thresholds
  -> release can continue
```

## Out Of Scope

- Starting Docker automatically.
- Load testing.
- Authentication.
- Canary traffic.
- Long-running soak test.
- Million request inference.

## Acceptance Criteria

- Smoke test script exists and executable.
- Script accepts base URL argument with default `http://127.0.0.1:8000`.
- Script fails fast on HTTP/API failure.
- Doc explains start compose → run smoke test → run production monitor.
- Tests assert script endpoints and docs commands.
- Existing tests stay green.

## Definition Of Done

`production-patterns domain` has production-like compose plus live API smoke verification path. Project covers train → gate → activate → serve → smoke test → monitor → alert → rollback → deploy metadata → CI.

## Next Step

[Step 26](./step-26-release-summary-report.md) adds release evidence summary report.
