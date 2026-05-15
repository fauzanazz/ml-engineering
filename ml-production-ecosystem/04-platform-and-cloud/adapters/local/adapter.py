"""Local provider adapter for filesystem and environment-based development."""

from dataclasses import dataclass
from pathlib import Path

from shared.platform import CloudResourceRef, InfrastructurePlan, SecretRef


@dataclass(frozen=True)
class LocalAdapterConfig:
    """Configuration references for local platform resources."""

    project_root: Path
    model_artifacts_path: str = "01-foundation/models"
    prediction_logs_path: str = "01-foundation/logs"
    registry_path: str = "01-foundation/registry/models.json"
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

    def _resource(self, name: str, relative_path: str) -> CloudResourceRef:
        return CloudResourceRef(
            provider=self.provider,
            kind="local-path",
            name=name,
            uri=str(self.config.project_root / relative_path),
        )
