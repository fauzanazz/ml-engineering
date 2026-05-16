# Step 24: Production Compose Profile

## Goal

Add Docker Compose production profile for `foundation-api` using deployment manifest conventions.

## User Story

Sebagai ML engineer, gw bisa start serving API dengan compose profile production, lalu endpoint health/metrics/drift tetap sama seperti release checklist dan deployment manifest.

## Command

Start production-like local serving:

```bash
docker compose -f docker-compose.production.yaml up --build foundation-api
```

Verify with monitor:

```bash
uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100
```

Stop:

```bash
docker compose -f docker-compose.production.yaml down
```

## Compose Shape

`docker-compose.production.yaml` defines `foundation-api` with:

- image/build aligned with `ml-production-ecosystem-foundation-api`
- command using `uv run foundation-serve-recommender`
- port `8000:8000`
- healthcheck against `/health`
- mounted local registry, artifacts, and logs paths

Endpoints stay aligned with [Step 21 deployment manifest](./step-21-deployment-manifest.md): `/health`, `/metrics`, `/metrics.json`, and `/drift`.

## Key Files

- `docker-compose.production.yaml`
- `docs/domains/production-patterns/production-compose.md`
- `configs/production-patterns/deploy/deployment-manifest.yaml`
- `tests/test_deployment_manifest.py`

## Pattern

```text
deployment manifest conventions
  -> docker-compose.production.yaml
  -> foundation-api local production-like runtime
  -> health/metrics/drift verification
```

## Out Of Scope

- Kubernetes.
- Cloud deployment.
- TLS/domain config.
- Autoscaling.
- Multi-replica serving.
- Real load balancer.
- Docker image push.

## Acceptance Criteria

- Production compose YAML exists and parses.
- Compose service name matches deployment manifest `foundation-api`.
- Port and endpoints align with manifest.
- Doc explains start, verify, stop flow.
- Tests assert compose config fields and doc commands.
- Existing tests stay green.

## Definition Of Done

`production-patterns domain` has local production-like runtime profile. Project covers train → gate → activate → serve → monitor → alert → rollback → deploy metadata → CI → production-like compose.

## Next Step

[Step 25](./step-25-live-api-smoke-test-script.md) adds live API smoke test against the running service.
