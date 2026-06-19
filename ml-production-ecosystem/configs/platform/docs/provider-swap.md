# Provider Swap Workflow

Purpose: move deployment targets without rewriting core ML lifecycle code.

## Rule

Core commands stay unchanged:

```bash
uv run production-lifecycle-demo --config configs/local-lifecycle-demo.yaml
uv run production-retrain --config configs/local-lifecycle-demo.yaml
uv run production-validate-offline --config configs/local-lifecycle-demo.yaml
uv run production-approve-model --validation-report artifacts/reports/production-patterns/offline-validation.json --approve
uv run production-rollback-model --registry-path registry/models.json --model-name <model> --target-version <version> --reason <reason>
```

Provider swap changes only these files:

```text
configs/platform/iac/<provider>/platform-plan.yaml
configs/platform/adapters/<provider>/
```

## Local Source Of Truth

Local comes first:

```bash
uv run production-apply-local-platform
uv run production-validate-platform-plan --plan-path configs/platform/iac/local/platform-plan.yaml
uv run production-validate-provider-portability
uv run production-provider-swap-matrix
```

The local adapter creates filesystem resources for artifacts, logs, and registry references. Cloud providers must match the same logical names through object storage, container registry, serving runtime, prediction logs, model registry, and secret references.

## Swap Steps

1. Keep model contract and lifecycle config stable.
2. Choose target provider plan under `configs/platform/iac/<provider>/platform-plan.yaml`.
3. Validate plan shape with `production-validate-platform-plan --plan-path configs/platform/iac/<provider>/platform-plan.yaml`.
4. Validate portability with `production-validate-provider-portability`.
5. Generate `production-provider-swap-matrix` and confirm core workflows require no code changes.
6. Implement or update thin provider adapter under `configs/platform/adapters/<provider>/`.
7. Preview deployment through adapter `deploy("development")` or tests, then run provider plan apply hooks with `--apply` after hook config is in place (`configs/platform/iac/<provider>/platform-plan.yaml`). Set `PLATFORM_APPLY_<PROVIDER>_COMMAND` for real cloud execution; without it, hooks emit safe fallback evidence.
8. Keep secret values outside repo; commit only `SecretRef` names, injection targets, and policy files under `configs/platform/policies/<provider>/`.

## Current Provider Matrix

| Provider | Current Proof | Runtime Status |
|---|---|---|
| local | Executable adapter + filesystem apply command | Runs without cloud credentials |
| AWS | Provider-neutral IaC plan shape + secret refs + plan-backed adapter | Dry-run preview + command hook evidence |
| GCP | Provider-neutral IaC plan shape + secret refs + plan-backed adapter | Dry-run preview + command hook evidence |
| Azure | Provider-neutral IaC plan shape + secret refs + plan-backed adapter | Dry-run preview + command hook evidence |

## Validation Commands

```bash
uv run production-validate-provider-boundaries
uv run production-validate-provider-portability
uv run production-provider-swap-matrix
uv run production-validate-secret-references
uv run production-validate-policy-references
```

These commands prove boundaries and portability at scaffold level. Adapter tests prove provider-neutral deployment preview. `--apply` mode executes command hooks declared in each provider plan and records command-level outcomes in deployment artifacts, including fallback-safe guard runs.
