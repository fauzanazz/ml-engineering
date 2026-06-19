# 04 Platform And Cloud

Purpose: show how production ML platform concerns attach to model-agnostic core workflows without coupling training, evaluation, serving, batch inference, monitoring, retraining, rollback, or release code to one vendor.

This stage is adapter-first. Provider examples are adapters, reference plans, dry-run deployment previews, and IaC module boundaries, not core architecture.

## Boundaries

| Layer | Owns | Must Not Own |
|---|---|---|
| `src/ml_production_ecosystem/shared/model_contracts` | Stable prediction, training, evaluation, and metadata contracts | Vendor SDK calls, project-specific recommender assumptions |
| `src/ml_production_ecosystem/shared/lifecycle` | Stable retraining, release, rollback, run, and report contracts | Scheduler, CI/CD, registry, or provider-specific behavior |
| `src/ml_production_ecosystem/shared/platform` | Provider-neutral resource, deployment execution, and secret references | Secret values, cloud SDK clients, Terraform implementation details |
| `adapters/local` | Local filesystem, Compose, and env-var injection examples | AWS/GCP/Azure behavior |
| `adapters/aws` | AWS-specific serving, storage, IAM, and secret wiring examples | Core model logic |
| `adapters/gcp` | GCP-specific serving, storage, IAM, and secret wiring examples | Core model logic |
| `adapters/azure` | Azure-specific serving, storage, IAM, and secret wiring examples | Core model logic |
| `iac/*` | Provider-specific infrastructure modules | Training/evaluation/serving business logic |

## Progressive Path

1. Keep `foundation docs/state` runnable locally with no cloud dependency.
2. Use `production-patterns domain` for lifecycle workflows: batch, monitoring, retraining, rollback, release summary.
3. Use `scale-reliability domain` for local reliability simulations before platform work.
4. Use this folder only when provider resources, IAM, managed secrets, CI/CD, or release environments are needed.

## Provider Rules

- Core code depends on `src/ml_production_ecosystem/shared/model_contracts`, `src/ml_production_ecosystem/shared/lifecycle`, `src/ml_production_ecosystem/shared/model_storage`, `src/ml_production_ecosystem/shared/deployment`, `src/ml_production_ecosystem/shared/monitoring`, `src/ml_production_ecosystem/shared/observability`, and `src/ml_production_ecosystem/shared/platform` contracts.
- Provider-specific imports stay inside `configs/platform/adapters/*` or `configs/platform/*`.
- Secret values never appear in code, docs, manifests, logs, or tests.
- Code may store secret names, IAM policy references, env-var names, and injection targets.
- New providers add adapters and IaC modules; they must not require rewriting core ML lifecycle code.

## Minimal Provider Matrix

| Provider | Artifact Storage | Serving | Batch | Secrets | IaC |
|---|---|---|---|---|---|
| Local | filesystem path | FastAPI/Compose | local JSONL job | `.env` reference only | Compose files |
| AWS | S3/ECR reference | ECS/SageMaker adapter | Batch/Lambda adapter | Secrets Manager/SSM reference | Terraform/CDK module |
| GCP | GCS/Artifact Registry reference | Cloud Run/Vertex adapter | Cloud Batch/Dataflow adapter | Secret Manager reference | Terraform module |
| Azure | Blob/ACR reference | Container Apps/ML endpoint adapter | Batch/Functions adapter | Key Vault reference | Terraform/Bicep module |

## Deliverables

- `src/ml_production_ecosystem/shared/model_contracts` defines model input/output and lifecycle ports.
- `src/ml_production_ecosystem/shared/lifecycle` defines retraining, release, and rollback workflow ports.
- `src/ml_production_ecosystem/shared/platform` defines provider, resource, deployment execution, and secret references.
- `shared.platform.PlatformPlanAdapter` loads any `iac/*/platform-plan.yaml` into the same `InfrastructurePlan` and `DeploymentExecution` contracts.
- `docs/provider-boundaries.md` defines allowed dependencies and adapter placement.
- `docs/secret-management.md` defines secret-reference-only patterns.
- `iac/*/platform-plan.yaml` files define provider-specific resource and secret references without secret values.

## Provider Swap

See [`docs/provider-swap.md`](docs/provider-swap.md) for local-first provider swap workflow and current local/AWS/GCP/Azure proof level.

## Policies As Code

Provider plans reference policy files under `policies/<provider>/`. Run `uv run production-validate-policy-references` to verify every `policy_ref` resolves without committing secret values.

## Local Secret Injection

Local secret references live under `secrets/local/secret-injections.yaml`. Run `uv run production-validate-local-secret-injections` to verify local env-var targets match the local platform plan without committing values.

## Local Kubernetes Parity

Local Kubernetes manifests live under `iac/local/kubernetes/`. Run `uv run production-validate-local-kubernetes` to verify kind/k3d-compatible deployment references without applying them or committing secret values.

## Local Scheduler

Local scheduler jobs live under `iac/local/scheduler/jobs.yaml`. Run `uv run production-validate-local-scheduler` to verify cron-compatible job definitions. Run `uv run production-run-local-scheduler --job-name lifecycle-status` to dry-run or `--execute` to run jobs without a managed scheduler.

## Cloud Provider Adapters

AWS, GCP, and Azure adapters are thin wrappers around provider-neutral platform plans. They expose `plan(...)` and dry-run `deploy(...)` previews through shared contracts without importing provider SDKs or reading credentials.

Runtime apply runs through command hooks declared in `configs/platform/iac/<provider>/platform-plan.yaml` using `production-apply-platform --apply ...`. Keep command templates reusable with placeholders (`{provider}`, `{environment}`, `{project_root}`). For controlled local-first execution, set a provider command through `PLATFORM_APPLY_<PROVIDER>_COMMAND` only when credentials/runtime are available; otherwise hooks use safe fallback and still emit command evidence.
