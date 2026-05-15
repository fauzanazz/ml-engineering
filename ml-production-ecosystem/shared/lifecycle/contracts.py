"""Provider-neutral contracts for ML lifecycle operations."""

from dataclasses import dataclass, field
from typing import Protocol

from shared.model_contracts import ModelMetadata


@dataclass(frozen=True)
class LifecycleRun:
    """One training, evaluation, monitoring, or release workflow run."""

    run_id: str
    workflow: str
    status: str
    model: ModelMetadata | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    report_uri: str | None = None


@dataclass(frozen=True)
class ReleaseDecision:
    """Provider-neutral release gate output."""

    approved: bool
    reason: str
    model: ModelMetadata
    checks: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RollbackPlan:
    """Provider-neutral rollback target and reason."""

    current_model: ModelMetadata
    target_model: ModelMetadata
    reason: str


class RetrainingPort(Protocol):
    """Boundary for scheduled or manual retraining workflows."""

    def run_retraining(self, config_uri: str) -> LifecycleRun:
        """Run retraining and return generic lifecycle result."""


class ReleasePort(Protocol):
    """Boundary for model release decisions."""

    def decide_release(self, run: LifecycleRun) -> ReleaseDecision:
        """Return release approval or rejection for workflow run."""


class RollbackPort(Protocol):
    """Boundary for model rollback operations."""

    def rollback(self, plan: RollbackPlan) -> LifecycleRun:
        """Execute rollback and return generic lifecycle result."""
