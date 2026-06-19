"""Production-pattern retraining entrypoint wrapping foundation config training."""

from pathlib import Path
import argparse
import json

import yaml

from ml_production_ecosystem.shared.model_storage.registry import register_model_version, set_active_model
from ml_production_ecosystem.recommendation.train import train_recommender_from_config
from .quality_gate import evaluate_quality_gate
from .training_adapters import TrainingAdapterError, run_training

DEFAULT_MODEL_NAME = "movielens-popularity"


def _load_config(config_path: Path) -> dict[str, object]:
    with config_path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        return {}
    return config


def _registry_path_from_config(config_path: Path) -> Path | None:
    config = _load_config(config_path)
    registry = config.get("registry", {})
    if not isinstance(registry, dict):
        return None
    registry_path = registry.get("path")
    if registry_path is None:
        return None
    return Path(str(registry_path))

def _run_training_from_config(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)
    training = config.get("training", {})
    if isinstance(training, dict) and training:
        try:
            return run_training(config_path=config_path, training=training)
        except TrainingAdapterError as error:
            raise ValueError(str(error)) from error

    result = train_recommender_from_config(config_path)
    return {
        "model_name": result.model_name,
        "version": result.version,
        "artifact_uri": result.uri,
        "metrics_uri": result.metrics_uri,
    }


def run_retraining(
    config_path: Path,
    set_active: bool = False,
    registry_path: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    require_quality_gate: bool = False,
) -> dict[str, object]:
    result = _run_training_from_config(config_path)
    config = _load_config(config_path)
    quality_gate_result = {"passed": True, "failures": []}
    if require_quality_gate:
        quality_gate = config.get("quality_gate", {})
        quality_gate_result = evaluate_quality_gate(
            quality_gate if isinstance(quality_gate, dict) else {},
            base=config_path.parent,
        )

    status = "completed"
    activated = False
    if set_active and quality_gate_result["passed"]:
        resolved_registry_path = registry_path or _registry_path_from_config(config_path)
        if resolved_registry_path is None:
            raise ValueError("--registry-path is required when --set-active and config has no registry.path")
        activation_model_name = model_name
        if model_name == DEFAULT_MODEL_NAME and str(result["model_name"]) != DEFAULT_MODEL_NAME:
            activation_model_name = str(result["model_name"])
        register_model_version(
            registry_path=resolved_registry_path,
            model_name=str(result["model_name"]),
            version=str(result["version"]),
            artifact_uri=str(result["artifact_uri"]),
            metrics_uri=str(result["metrics_uri"]),
        )
        set_active_model(resolved_registry_path, activation_model_name, str(result["version"]))
        activated = True
    elif set_active and not quality_gate_result["passed"]:
        status = "failed_quality_gate"

    return {
        "status": status,
        "model_name": str(result["model_name"]),
        "version": str(result["version"]),
        "artifact_uri": str(result["artifact_uri"]),
        "metrics_uri": str(result["metrics_uri"]),
        "set_active": activated,
        "quality_gate": quality_gate_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production-pattern retraining from foundation config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set-active", action="store_true")
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--require-quality-gate", action="store_true")
    args = parser.parse_args()

    summary = run_retraining(args.config, args.set_active, args.registry_path, args.model_name, args.require_quality_gate)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
