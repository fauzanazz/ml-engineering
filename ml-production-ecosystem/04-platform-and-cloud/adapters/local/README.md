# local Adapter

Scope: keep local-specific runtime integration behind shared contracts.

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

`adapter.py` implements `LocalProviderAdapter`, which returns filesystem resources and env-var secret references through `shared.platform.InfrastructurePlan`. It can also create required local directories with `ensure_resources()` so the local stack is executable, not only descriptive.

Run it through `uv run production-apply-local-platform`. It does not read cloud credentials, resolve secret values, or import provider SDKs.
