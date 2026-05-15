# aws Adapter

Scope: keep aws-specific runtime integration behind shared contracts.

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

Adapter implementations must expose shared ports from `shared/model_contracts`, `shared/deployment`, `shared/model_storage`, `shared/monitoring`, `shared/observability`, or `shared/platform`.
