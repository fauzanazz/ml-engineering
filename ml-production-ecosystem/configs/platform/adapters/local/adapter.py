"""Local provider adapter for filesystem and environment-based development."""

from dataclasses import dataclass
from pathlib import Path

from ml_production_ecosystem.shared.platform import CloudResourceRef, InfrastructurePlan, SecretRef

@dataclass(frozen=True)
class LocalAdapterConfig:
    """Configuration references for local platform resources."""

    project_root: Path
    model_artifacts_path: str = "artifacts/foundation"
    prediction_logs_path: str = "logs"
    registry_path: str = "registry/models.json"
    registry_token_env: str = "LOCAL_MODEL_REGISTRY_TOKEN"

class LocalProviderAdapter:
    """Provider adapter for local-first workflows without cloud credentials."""

    provider = "local"

    def __init__(self, config: LocalAdapterConfig) -> None:
        self.config = config

    def plan(self, environment: str) -> InfrastructurePlan:
        """Return local resource and secret references only."""

        return InfrastructurePlan(
            provider=self.provider,
            resources=(
                self._resource("model-artifacts", self.config.model_artifacts_path),
                self._resource("prediction-logs", self.config.prediction_logs_path),
                self._resource("model-registry", self.config.registry_path),
            ),
            secrets=(
                SecretRef(
                    provider=self.provider,
                    name=f"{environment}/local/model-registry-token",
                    injection_target=self.config.registry_token_env,
                    policy_ref="local-env-file-reference",
                ),
            ),
        )

    def ensure_resources(self, environment: str) -> dict[str, object]:
        """Create local filesystem resources referenced by the plan."""

        plan = self.plan(environment)
        resources = []
        for resource in plan.resources:
            path = Path(resource.uri)
            if resource.name == "model-registry":
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
            resources.append(
                {
                    "name": resource.name,
                    "kind": resource.kind,
                    "uri": resource.uri,
                    "ready": path.parent.exists() if resource.name == "model-registry" else path.exists(),
                }
            )
        return {
            "status": "ready" if all(resource["ready"] for resource in resources) else "incomplete",
            "provider": self.provider,
            "environment": environment,
            "resources": resources,
            "secret_references": [secret.injection_target for secret in plan.secrets],
        }

    def _resource(self, name: str, relative_path: str) -> CloudResourceRef:
        return CloudResourceRef(
            provider=self.provider,
            kind="local-path",
            name=name,
            uri=str(self.config.project_root / relative_path),
        )
