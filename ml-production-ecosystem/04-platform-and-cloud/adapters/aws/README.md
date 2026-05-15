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

## Current Adapter

`adapter.py` implements a thin plan-backed `aws` provider adapter using `shared.platform.PlatformPlanAdapter`. It returns provider resource and secret references from `04-platform-and-cloud/iac/aws/platform-plan.yaml` without importing provider SDKs or reading cloud credentials.
