"""Training adapters for model-agnostic lifecycle execution.

This module centralizes non-core training execution and exposes
framework seams for ONNX and PyTorch command-based workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import json
import subprocess


class TrainingAdapterError(ValueError):
    """Raised when training config or command summary is invalid."""


@dataclass(frozen=True)
class TrainingSummary:
    """Normalized training summary artifact."""

    model_name: str
    version: str
    artifact_uri: str
    metrics_uri: str

    def as_dict(self) -> dict[str, str]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "artifact_uri": self.artifact_uri,
            "metrics_uri": self.metrics_uri,
        }


REQUIRED_SUMMARY_KEYS = ("model_name", "version", "artifact_uri", "metrics_uri")
SUPPORTED_TRAINING_TYPES = ("command", "onnx", "pytorch", "transformers", "xgboost", "sklearn", "external")


class TrainingAdapter(Protocol):
    """Boundary for model-training execution strategies."""

    def run(self, config_path: Path, training: Mapping[str, object]) -> TrainingSummary:
        """Run training and return normalized summary."""


def _coerce_path(config_path: Path, value: object) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else config_path.parent / candidate


def _resolve_command(training: Mapping[str, object]) -> list[str]:
    command = training.get("command")
    if not isinstance(command, list) or not command:
        raise TrainingAdapterError("training.command must be a non-empty list")
    resolved: list[str] = [str(part) for part in command]
    if any(part == "" for part in resolved):
        raise TrainingAdapterError("training.command items cannot be empty")
    return resolved


def _read_training_summary(summary_path: Path) -> TrainingSummary:
    try:
        payload = json.loads(summary_path.read_text())
    except FileNotFoundError as exc:
        raise TrainingAdapterError(f"training.summary_path not found: {summary_path}") from exc
    except json.JSONDecodeError as exc:
        raise TrainingAdapterError(f"training.summary_path must contain valid JSON: {summary_path}") from exc

    if not isinstance(payload, dict):
        raise TrainingAdapterError("training summary JSON must be an object")

    missing = [key for key in REQUIRED_SUMMARY_KEYS if key not in payload]
    if missing:
        raise TrainingAdapterError(f"training summary missing required fields: {', '.join(missing)}")

    summary = TrainingSummary(
        model_name=str(payload["model_name"]),
        version=str(payload["version"]),
        artifact_uri=str(payload["artifact_uri"]),
        metrics_uri=str(payload["metrics_uri"]),
    )

    for key, value in (
        ("model_name", summary.model_name),
        ("version", summary.version),
        ("artifact_uri", summary.artifact_uri),
        ("metrics_uri", summary.metrics_uri),
    ):
        if not value:
            raise TrainingAdapterError(f"training summary field '{key}' must be non-empty")

    return summary


class CommandTrainingAdapter:
    """Execute external command training and parse deterministic summary JSON."""

    def run(self, config_path: Path, training: Mapping[str, object]) -> TrainingSummary:
        summary_path = training.get("summary_path")
        if summary_path is None:
            raise TrainingAdapterError("training.summary_path is required for command-based training")

        command = _resolve_command(training)
        subprocess.run(command, check=True, cwd=config_path.parent)

        summary_file = _coerce_path(config_path, summary_path)
        return _read_training_summary(summary_file)


class FrameworkCommandAdapter:
    """Framework-specific adapter wrapper on top of command execution."""

    def __init__(self, framework: str, command_adapter: CommandTrainingAdapter | None = None) -> None:
        self.framework = framework
        self.command_adapter = command_adapter or CommandTrainingAdapter()

    def run(self, config_path: Path, training: Mapping[str, object]) -> TrainingSummary:
        normalized_training: dict[str, object] = {"framework": self.framework}
        normalized_training.update(training)
        return self.command_adapter.run(config_path, normalized_training)


class OnnxTrainingAdapter(FrameworkCommandAdapter):
    """Command adapter explicit seam for ONNX-first trainers."""

    def __init__(self, command_adapter: CommandTrainingAdapter | None = None) -> None:
        super().__init__(framework="onnx", command_adapter=command_adapter)


class PyTorchTrainingAdapter(FrameworkCommandAdapter):
    """Command adapter explicit seam for PyTorch-first trainers."""

    def __init__(self, command_adapter: CommandTrainingAdapter | None = None) -> None:
        super().__init__(framework="pytorch", command_adapter=command_adapter)


def _resolve_adapter(training: Mapping[str, object]) -> TrainingAdapter:
    training_type = str(training.get("type", "command"))
    if training_type not in SUPPORTED_TRAINING_TYPES:
        known = ", ".join(SUPPORTED_TRAINING_TYPES)
        raise TrainingAdapterError(f"unsupported training.type '{training_type}'. Supported: {known}")

    if training_type == "command":
        return CommandTrainingAdapter()
    if training_type == "onnx":
        return OnnxTrainingAdapter()
    if training_type == "pytorch":
        return PyTorchTrainingAdapter()

    return FrameworkCommandAdapter(training_type)


def run_training(config_path: Path, training: Mapping[str, object]) -> dict[str, object]:
    adapter = _resolve_adapter(training)
    summary = adapter.run(config_path, training)
    return summary.as_dict()
