# local IaC Module

Scope: define local-specific infrastructure as code without changing core ML workflows.

Allowed here:

- compute, storage, network, IAM, and scheduler resources
- secret container names and policy references
- outputs consumed by provider adapters

Forbidden here:

- secret values
- model training code
- model serving business logic
- assumptions that prevent another provider adapter from implementing same contract

## Current Plan

`platform-plan.yaml` defines local filesystem resources and env-var secret references as code. Values must be supplied outside git by local environment or ignored `.env` files.
