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
7. Preview deployment through `uv run production-apply-platform --provider <provider> --project-root .` (dry-run default), then execute runtime apply hooks via `uv run production-apply-platform --provider <provider> --apply --project-root .` when command hooks are configured in `configs/platform/iac/<provider>/platform-plan.yaml`. Add placeholders in `apply.commands` such as `{provider}`, `{environment}`, `{project_root}` for reusable command templates.
8. For real cloud execution, set `PLATFORM_APPLY_<PROVIDER>_COMMAND` (contoh `PLATFORM_APPLY_AWS_COMMAND`) with a shell-safe command string. The runtime receives `{provider}`, `{environment}`, `{project_root}` replacements.
9. Keep secret values outside repo; commit only `SecretRef` names, injection targets, and policy files under `configs/platform/policies/<provider>/`. 

## Current Provider Matrix

| Provider | Current Proof | Runtime Status |
|---|---|---|
| local | Executable adapter + filesystem apply command | Runs without cloud credentials |
| AWS | Provider-neutral IaC plan shape + secret refs + plan-backed adapter | Dry-run preview + runtime apply command hooks |
| GCP | Provider-neutral IaC plan shape + secret refs + plan-backed adapter | Dry-run preview + runtime apply command hooks |
| Azure | Provider-neutral IaC plan shape + secret refs + plan-backed adapter | Dry-run preview + runtime apply command hooks |

## Validation Commands

```bash
uv run production-validate-provider-boundaries
uv run production-validate-provider-portability
uv run production-provider-swap-matrix
uv run production-validate-secret-references
uv run production-validate-policy-references
```

These commands prove boundaries and portability at scaffold level. `production-apply-platform` emits structured execution reports for local and cloud plans. Dry-run keeps adapter outputs non-destructive; `--apply` runs command hooks from the platform plan and emits per-command dispatch evidence. If a provider command is absent, hooks still run fallback-safe output and keep audit evidence intact.
