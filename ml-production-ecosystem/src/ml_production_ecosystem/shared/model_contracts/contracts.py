"""Stable contracts for model-agnostic ML workflows."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
FeatureMap = Mapping[str, JsonValue]


@dataclass(frozen=True)
class ModelIOContract:
    """Reusable schema reference for model inputs and outputs."""

    input_schema_uri: str
    output_schema_uri: str
    task_type: str
    prediction_key: str


@dataclass(frozen=True)
class ModelMetadata:
    """Model identity without provider or framework assumptions."""

    name: str
    version: str
    framework: str
    io_contract: ModelIOContract
    artifact_uri: str | None = None


@dataclass(frozen=True)
class PredictionRequest:
    """Generic prediction input accepted by online and batch adapters."""

    records: Sequence[FeatureMap]
    request_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionResponse:
    """Generic prediction output returned by any model type."""

    predictions: Sequence[JsonValue]
    model: ModelMetadata
    request_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEnvelope:
    """Loaded model plus metadata behind framework-specific adapters."""

    model: Any
    metadata: ModelMetadata


class PredictionPort(Protocol):
    """Boundary for online prediction implementations."""

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Return predictions using generic request and response contracts."""


class BatchPredictionPort(Protocol):
    """Boundary for batch prediction implementations."""

    def predict_batch(self, input_uri: str, output_uri: str, model: ModelMetadata) -> str:
        """Run batch predictions and return output URI."""


class TrainingPort(Protocol):
    """Boundary for training implementations."""

    def train(self, config_uri: str, output_uri: str) -> ModelMetadata:
        """Train model and return provider-neutral metadata."""


class EvaluationPort(Protocol):
    """Boundary for evaluation implementations."""

    def evaluate(self, model: ModelMetadata, dataset_uri: str) -> Mapping[str, float]:
        """Evaluate model and return metric names with values."""
