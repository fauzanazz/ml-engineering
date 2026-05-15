# gcp IaC Module

Scope: define gcp-specific infrastructure as code without changing core ML workflows.

Allowed here:

- compute, storage, network, IAM, and scheduler resources
- secret container names and policy references
- outputs consumed by provider adapters

Forbidden here:

- secret values
- model training code
- model serving business logic
- assumptions that prevent another provider adapter from implementing same contract
