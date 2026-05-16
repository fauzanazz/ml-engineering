# azure Adapter

Scope: keep azure-specific runtime integration behind shared contracts.

Allowed here:

- provider SDK clients
- provider payload mapping
- provider endpoint and job naming
- provider auth lookup by secret reference

Forbidden here:

- core training logic
- core evaluation logic
- model-specific business assumptions
- committed secret values

Adapter implementations must expose shared ports from `src/ml_production_ecosystem/shared/model_contracts`, `src/ml_production_ecosystem/shared/deployment`, `src/ml_production_ecosystem/shared/model_storage`, `src/ml_production_ecosystem/shared/monitoring`, `src/ml_production_ecosystem/shared/observability`, or `src/ml_production_ecosystem/shared/platform`.

## Current Adapter

`adapter.py` implements a thin plan-backed `azure` provider adapter using `shared.platform.PlatformPlanAdapter`. It returns provider resource and secret references from `configs/platform/azure/platform-plan.yaml` without importing provider SDKs or reading cloud credentials.
