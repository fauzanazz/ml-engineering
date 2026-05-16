"""Model-agnostic input, output, and lifecycle contracts."""

from .contracts import (
    BatchPredictionPort,
    EvaluationPort,
    ModelEnvelope,
    ModelIOContract,
    ModelMetadata,
    PredictionPort,
    PredictionRequest,
    PredictionResponse,
    TrainingPort,
)

__all__ = [
    "BatchPredictionPort",
    "EvaluationPort",
    "ModelEnvelope",
    "ModelIOContract",
    "ModelMetadata",
    "PredictionPort",
    "PredictionRequest",
    "PredictionResponse",
    "TrainingPort",
]
