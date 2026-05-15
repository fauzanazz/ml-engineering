---
title: Security Controls
type: operational-doc
created: 2026-05-15
status: completed
categories: [ml-production, security, compliance]
related:
  - ./deployment-manifest.md
  - ./release-checklist.md
  - ./incident-simulation.md
  - ../../docs/features/step-7-prediction-logging-and-basic-drift-signal.md
---

# Security Controls

This document records baseline security controls for the local production ML pattern. Current implementation is local-only, but controls are written so the same reasoning can move to ECS, EKS, SageMaker, or another production platform.

## Security Goals

- Keep model artifacts and registry state protected from accidental overwrite.
- Avoid secrets in source code, logs, Docker images, and docs.
- Keep prediction logs privacy-safe.
- Make release, rollback, and incident activity auditable.
- Use least privilege for every runtime, job, and operator path.

## Data Classification

| Data | Example | Sensitivity | Control |
|---|---|---|---|
| Source code | Python modules, docs, manifests | low | review before merge |
| Model artifact | trained recommender artifact | medium | versioned storage, write restrictions |
| Registry metadata | `01-foundation/registry/models.json` | medium | controlled activation/rollback path |
| Prediction logs | JSONL request/response metadata | medium/high if user identifiers appear | minimize fields, avoid raw PII |
| Metrics | latency, error count, drift score | low/medium | expose operational aggregates only |
| Secrets | API keys, cloud credentials | high | never commit; use secret manager/env injection |

## Least Privilege

Local project controls:

- Training command writes only to configured artifact, run, and registry paths.
- Serving API reads active model metadata and model artifacts; it should not mutate registry state during normal prediction.
- Retraining and rollback CLIs are the only paths that should change active model state.
- Monitoring command reads API endpoints only; it should not write model artifacts or registry state.

Cloud mapping:

| Component | Minimum permissions |
|---|---|
| serving runtime | read model artifact, read registry pointer, write logs/metrics |
| training job | read training data, write candidate artifact, write run metadata |
| activation job | read candidate metrics, update active model pointer |
| rollback job | read registry history, update active model pointer |
| monitor | read health/metrics/drift endpoints, write alert/check result |
| operator | trigger approved runbooks, view logs, no broad admin by default |

Avoid broad policies such as:

```text
s3:*
dynamodb:*
secretsmanager:*
AdministratorAccess
```

Prefer resource-scoped permissions and separate roles for serving, training, monitoring, activation, and rollback.

## Secrets Handling

Rules:

- Do not commit `.env`, cloud credentials, tokens, private keys, or service account JSON.
- Do not print secrets in CLI output, exception messages, logs, or docs.
- Use environment variables only as injection mechanism, not as long-term secret storage.
- In AWS, use Secrets Manager or SSM Parameter Store with IAM task/job role access.
- In Kubernetes, use Secret objects backed by a cloud secret store when possible.

Local placeholder pattern:

```text
SECRET_VALUE=<provided by runtime secret manager>
```

Never use realistic secret values in examples.

## Privacy-Safe Logging

Prediction logs should support debugging and drift checks without storing unnecessary personal data.

Allowed by default:

- request timestamp
- model name/version
- latency
- status/error category
- aggregate input shape
- top-level validation failure reason
- anonymized or hashed correlation ID if needed

Avoid by default:

- raw user identifiers
- account numbers
- email addresses
- phone numbers
- names
- free-text user input
- full request payloads
- access tokens or cookies

If identifiers are required for debugging, use short retention, explicit approval, and irreversible hashing where possible.

## Audit Trail

Release, rollback, and incident actions should produce durable evidence:

- scheduled retraining report: `02-production-patterns/reports/scheduled-retraining.json`
- rollback report: `02-production-patterns/reports/rollback.json`
- monitor output from `production-monitor`
- incident record from [incident-simulation.md](./incident-simulation.md)
- active model version before and after change
- operator and reason for activation/rollback

Minimum audit fields:

```text
timestamp
operator_or_job
command_or_workflow
model_name
source_version
target_version
reason
quality_gate_status
monitor_status
rollback_available
```

## Model Artifact Integrity

Controls:

- Treat artifacts as immutable after registration.
- Register new versions instead of overwriting existing ones.
- Keep known-good versions available for rollback.
- Validate artifact path before activation.
- Record training config and metrics with candidate version.
- Block activation when quality gate fails.

Future cloud controls:

- S3 bucket versioning.
- Server-side encryption.
- Object lock or write-once policy for promoted artifacts.
- Checksum validation before loading artifact.
- Separate write role for training and read role for serving.

## Network And Runtime Controls

Local controls:

- Bind local testing to `127.0.0.1` unless container publishing requires `0.0.0.0` inside container.
- Expose only required service port `8000` in production compose.
- Keep `/metrics` and `/metrics.json` operational only; do not include secrets or raw payloads.

Cloud controls:

- Put serving API behind load balancer/API gateway.
- Use TLS at ingress.
- Restrict admin endpoints from public access.
- Use private subnets for internal services when possible.
- Emit logs and metrics to centralized audit/observability system.

## Compliance-Aware Defaults

This project is not compliance-certified. It should still show compliance-aware habits:

- Document what data is logged and why.
- Minimize stored prediction data.
- Retain only operationally useful logs.
- Separate duties between training, serving, activation, and rollback.
- Preserve evidence for releases and incidents.
- Review access before moving from local to cloud.

## Security Review Checklist

Before release:

- [ ] No secrets committed or copied into docs.
- [ ] Serving path does not mutate registry during prediction.
- [ ] Activation requires quality gate pass.
- [ ] Rollback target is known and still registered.
- [ ] Prediction logs do not include raw PII.
- [ ] Metrics endpoints expose aggregates only.
- [ ] Operator records model version, reason, and monitor status.

Before cloud deployment:

- [ ] Runtime roles are split by responsibility.
- [ ] Artifact storage has encryption and versioning.
- [ ] Secret manager is used for runtime secrets.
- [ ] Public ingress has TLS.
- [ ] Logs have retention policy.
- [ ] Audit events include release and rollback actions.
