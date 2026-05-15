# Provider Boundaries

Production ML code should be model-agnostic first and provider-agnostic second.

## Allowed Dependency Direction

```text
application workflows
  -> shared contracts
  -> provider adapter interface
  -> provider-specific adapter
  -> provider SDK or IaC module
```

Core training, evaluation, serving, batch inference, monitoring, retraining, rollback, and release workflows must not import AWS, GCP, Azure, Kubernetes, Terraform, or vendor SDK modules directly.

## Adapter Contract

Adapters translate generic contracts into provider-specific behavior:

- `PredictionRequest` and `PredictionResponse` map to serving payloads.
- `ModelMetadata` maps to artifact registry, container image, or endpoint references.
- `LifecycleRun`, `ReleaseDecision`, and `RollbackPlan` map to scheduler, CI/CD, and registry actions.
- `InfrastructurePlan` maps to provider IaC outputs.
- `SecretRef` maps to provider-managed secret names and injection targets.

## Review Gate

Before adding vendor code, answer:

1. Can local workflow still run without provider credentials?
2. Does new provider code live under `04-platform-and-cloud/adapters/*` or `04-platform-and-cloud/iac/*`?
3. Does core code depend only on shared contracts?
4. Are secret values excluded from code and tests?
5. Can another provider implement same behavior through new adapter only?
