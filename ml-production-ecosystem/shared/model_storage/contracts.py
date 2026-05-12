"""Model artifact storage contracts shared across learning stages."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelArtifact:
    """Metadata for a stored model artifact."""

    name: str
    version: str
    uri: str
    metrics_uri: str | None = None


class ModelStore(Protocol):
    """Boundary for model artifact storage and retrieval."""

    def register(self, artifact: ModelArtifact) -> None:
        """Register model artifact metadata."""

    def get(self, name: str, version: str) -> ModelArtifact:
        """Return model artifact metadata by name and version."""
