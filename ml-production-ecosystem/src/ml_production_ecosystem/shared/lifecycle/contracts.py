"""Provider-neutral contracts for ML lifecycle operations."""

from dataclasses import dataclass, field
from typing import Protocol

from ml_production_ecosystem.shared.model_contracts import ModelMetadata


@dataclass(frozen=True)
class DatasetRef:
    """Provider-neutral dataset reference consumed by training workflows."""

    name: str
    uri: str
    schema_uri: str
    version: str | None = None


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


@dataclass(frozen=True)
class OfflineValidationReport:
    """Offline model validation result before deployment approval."""

    model: ModelMetadata
    dataset: DatasetRef
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    report_uri: str | None = None


@dataclass(frozen=True)
class DeploymentDemoResult:
    """Smoke-test result for a local or provider deployment demo."""

    endpoint: str
    passed: bool
    checks: tuple[str, ...] = field(default_factory=tuple)
    report_uri: str | None = None


@dataclass(frozen=True)
class DriftReport:
    """Model/data drift signal independent of monitoring backend."""

    model: ModelMetadata
    score: float
    threshold: float
    passed: bool
    sample_size: int
    report_uri: str | None = None


@dataclass(frozen=True)
class ContinualLearningDecision:
    """Decision that connects monitoring evidence to retraining action."""

    action: str
    reason: str
    trigger: str
    approved_for_retraining: bool


@dataclass(frozen=True)
class LifecycleGraph:
    """Interactive or renderable lifecycle flow definition."""

    name: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    renderer: str = "mermaid"


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


class DataIngestionPort(Protocol):
    """Boundary for adding or validating datasets."""

    def ingest(self, source_uri: str, schema_uri: str) -> DatasetRef:
        """Return dataset reference after ingestion or validation."""


class OfflineValidationPort(Protocol):
    """Boundary for validating model quality before deployment."""

    def validate_offline(self, model: ModelMetadata, dataset: DatasetRef) -> OfflineValidationReport:
        """Return offline quality report."""


class ApprovalPort(Protocol):
    """Boundary for manual or automated model approval."""

    def approve(self, report: OfflineValidationReport) -> ReleaseDecision:
        """Return deployment approval decision."""


class DeploymentDemoPort(Protocol):
    """Boundary for deployment smoke tests and demos."""

    def run_demo(self, endpoint: str) -> DeploymentDemoResult:
        """Run deployment demo checks and return result."""


class DriftDetectionPort(Protocol):
    """Boundary for drift detection implementations."""

    def detect_drift(self, model: ModelMetadata, dataset: DatasetRef) -> DriftReport:
        """Return drift report for model and observed dataset."""


class ContinualLearningPort(Protocol):
    """Boundary for monitoring-to-retraining decisions."""

    def decide(self, drift: DriftReport) -> ContinualLearningDecision:
        """Return continual-learning decision from monitoring evidence."""


class LifecycleGraphPort(Protocol):
    """Boundary for graph-based lifecycle flow rendering."""

    def render_graph(self, graph: LifecycleGraph) -> str:
        """Render graph definition for UI or docs."""
