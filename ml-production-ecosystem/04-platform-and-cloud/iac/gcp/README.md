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

## Current Plan

`platform-plan.yaml` defines GCP resource references for artifacts, images, serving, prediction logs, model registry, and model-registry secret injection. It is a reference contract only: no Terraform state, credentials, project secrets, or secret values are committed.
