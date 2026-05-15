# Secret Management

This project commits secret references, policies-as-code, and injection patterns only. It never commits secret values.

## Allowed In Code

- Secret names such as `prod/ml/api-token`.
- Environment variable targets such as `MODEL_REGISTRY_TOKEN`.
- IAM or policy references such as `policy/ml-serving-read-secret`.
- Provider locations such as AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, or local `.env` file path references.

## Forbidden In Code

- API keys, tokens, passwords, private keys, database credentials, signing secrets, or connection strings with credentials.
- Base64-encoded secret values.
- Test fixtures containing real or realistic secret values.
- Logs that print resolved secret values.

## Injection Pattern

1. IaC creates or references external secret containers.
2. Deployment manifest references secret name and policy.
3. Runtime adapter injects value into env var or mounted file.
4. Core ML code reads configured env var or file path without knowing provider.
5. Tests assert references exist, not values.

## Local Enforcement

Run these before push when changing platform plans, policy refs, or configs:

```bash
production-validate-secret-references
production-validate-local-secret-injections
production-validate-policy-references
```

The validator scans committed YAML/JSON under `configs` and `04-platform-and-cloud` for forbidden secret-value keys such as `api_key`, `password`, `private_key`, `secret_value`, and `token_value`.

## Policy References

Every `policy_ref` in `04-platform-and-cloud/iac/*/platform-plan.yaml` must resolve to a YAML file under `04-platform-and-cloud/policies/<provider>/`. Policy files define provider, policy ref, effect, actions, resources, injection targets, and `value_handling: external-only`.

## Local Secret Injection Manifest

`04-platform-and-cloud/secrets/local/secret-injections.yaml` lists required env-var targets and policy references for local runs. It never stores secret values; `value_handling: external-only` means users provide values outside git through their shell, ignored `.env`, or local secret tooling.
