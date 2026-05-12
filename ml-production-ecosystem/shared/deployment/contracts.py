"""Model deployment contracts shared across learning stages."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DeploymentResult:
    """Result returned after model deployment step completes."""

    model_name: str
    model_version: str
    endpoint: str


class ModelDeployer(Protocol):
    """Boundary for components that deploy model artifacts."""

    def deploy(self, model_name: str, model_version: str, artifact_uri: str) -> DeploymentResult:
        """Deploy model artifact and return endpoint metadata."""
