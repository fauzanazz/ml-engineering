# Step 21: Deployment Manifest

## Goal

Add deployment manifest doc/config describing how local `foundation-api` is released as a named service with image, command, ports, health, metrics, drift, registry, release checklist, and rollback notes.

## User Story

Sebagai ML engineer, gw bisa lihat “model serving service” ini dideploy seperti apa: service name, command, endpoint, healthcheck, metrics endpoint, dan release checklist link.

## Manifest Fields

`02-production-patterns/deploy/deployment-manifest.yaml` records:

| Field | Value |
|---|---|
| `service_name` | `foundation-api` |
| `image` | `ml-production-ecosystem-foundation-api` |
| `command` | `uv run foundation-serve-recommender ...` |
| `port` | `8000` |
| `health_endpoint` | `/health` |
| `metrics_endpoint` | `/metrics` |
| `metrics_json_endpoint` | `/metrics.json` |
| `drift_endpoint` | `/drift` |
| `registry_path` | `01-foundation/registry/models.json` |
| `release_checklist` | `02-production-patterns/docs/release-checklist.md` |
| `rollback_command` | `uv run production-rollback-model ...` |

Companion docs live in `02-production-patterns/docs/deployment-manifest.md` and explain each field plus when to update it.

## Key Files

- `02-production-patterns/deploy/deployment-manifest.yaml`
- `02-production-patterns/docs/deployment-manifest.md`
- `tests/test_deployment_manifest.py`

## Pattern

```text
release checklist
  -> deployment manifest
  -> service name/image/command/endpoints
  -> monitor and rollback metadata
```

## Out Of Scope

- Kubernetes YAML.
- Terraform.
- Helm chart.
- CI/CD deploy job.
- Cloud load balancer.
- Autoscaling.

## Acceptance Criteria

- Deployment manifest YAML exists and parses.
- Manifest references serving command, port, endpoints, registry, release checklist, rollback command.
- Doc explains each field and when to update it.
- Tests assert manifest required fields and endpoint values.
- Existing tests stay green.

## Definition Of Done

`02-production-patterns` has release checklist plus deployment manifest. Project covers train → gate → activate → serve → monitor → alert → rollback → deploy metadata.

## Next Step

[Step 22](./step-22-lightweight-ci-validation.md) adds local CI validation.
