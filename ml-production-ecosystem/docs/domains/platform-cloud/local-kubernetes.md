# Local Kubernetes Manifests

Purpose: provide kind/k3d-compatible local orchestration manifests while keeping local stack source of truth and cloud providers optional adapters.

These manifests are reference-only unless user chooses to run a local cluster. They do not require cloud credentials and do not commit secret values.

Validate:

```bash
uv run production-validate-local-kubernetes
```

Runtime boundary: `Secret` values must be created outside git. Manifest uses `secretKeyRef` only.
