# Secret Management

This project commits secret references, policies, and injection patterns only. It never commits secret values.

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
