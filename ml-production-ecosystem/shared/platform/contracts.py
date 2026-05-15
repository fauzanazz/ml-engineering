"""Thin contracts for provider-specific infrastructure adapters."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SecretRef:
    """Reference to externally managed secret value."""

    provider: str
    name: str
    injection_target: str
    policy_ref: str


@dataclass(frozen=True)
class CloudResourceRef:
    """Provider-neutral reference to infrastructure resource."""

    provider: str
    kind: str
    name: str
    uri: str


@dataclass(frozen=True)
class InfrastructurePlan:
    """Provider adapter output consumed by release workflows."""

    provider: str
    resources: tuple[CloudResourceRef, ...] = field(default_factory=tuple)
    secrets: tuple[SecretRef, ...] = field(default_factory=tuple)


class ProviderAdapter(Protocol):
    """Boundary for local, AWS, GCP, Azure, or future providers."""

    provider: str

    def plan(self, environment: str) -> InfrastructurePlan:
        """Return resource and secret references for environment."""
