---
title: Foundation API Deployment Manifest
type: feature-note
created: 2026-05-14
status: completed
categories: [ml-production, deployment, serving]
related:
  - ../deploy/deployment-manifest.yaml
  - ./release-checklist.md
  - ./alerting-runbook.md
---

# Foundation API Deployment Manifest

`deployment-manifest.yaml` names the local model serving service and records the operational endpoints needed for release, monitoring, alerting, and rollback.

## Manifest Fields

`service_name` identifies the serving service as `foundation-api`. Keep this aligned with Docker Compose service names and monitoring dashboards.

`image` records the expected local image name, `ml-production-ecosystem-foundation-api`. Update it when packaging or image naming changes.

`command` records the serving command, `uv run foundation-serve-recommender`. Update it when the API entrypoint changes.

`port` records the service port, `8000`. Update it when the API binds to a different exposed port.

`health_endpoint` records `/health`, the readiness check used by manual verification and alerting.

`metrics_endpoint` records `/metrics`, the Prometheus-style metrics endpoint used by the monitoring stack.

`metrics_json_endpoint` records `/metrics.json`, the machine-readable metrics endpoint used by `production-monitor`.

`drift_endpoint` records `/drift`, the model drift endpoint used after activation and during release verification.

`registry_path` records `01-foundation/registry/models.json`, the local registry file that stores registered versions and active model state.

`release_checklist` points to [release-checklist.md](./release-checklist.md), the manual release workflow for train, quality gate, activation, monitoring, alert checks, and rollback.

`rollback_command` records the emergency command using [production rollback](../production_patterns/rollback.py). Replace `--target-version` with the known-good version captured before release.

## When to update

Update `02-production-patterns/deploy/deployment-manifest.yaml` whenever service name, image name, command, port, endpoint path, registry path, release checklist path, or rollback procedure changes.

Also update this document when new manifest fields are added, because operators should be able to understand every field without reading code.

## Release usage

Before release, compare manifest values against the running service. During release, follow [release-checklist.md](./release-checklist.md). After release, use [alerting-runbook.md](./alerting-runbook.md) if `production-monitor` reports an unhealthy state.
