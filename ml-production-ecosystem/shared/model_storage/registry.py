"""Generic local model registry shared across model types."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {"models": [], "active": {}}
    with registry_path.open() as file:
        registry = json.load(file)
    registry.setdefault("models", [])
    registry.setdefault("active", {})
    return registry


def _write(registry_path: Path, registry: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))


def register_model_version(
    registry_path: Path,
    model_name: str,
    version: str,
    artifact_uri: str,
    metrics_uri: str,
    stage: str = "candidate",
    set_active: bool = False,
) -> dict[str, Any]:
    registry = _load(registry_path)
    models = registry["models"]
    models[:] = [
        model
        for model in models
        if not (model.get("model_name") == model_name and model.get("version") == version)
    ]
    entry = {
        "model_name": model_name,
        "version": version,
        "artifact_uri": artifact_uri,
        "metrics_uri": metrics_uri,
        "stage": stage,
        "created_at": datetime.now(UTC).isoformat(),
    }
    models.append(entry)
    if set_active:
        registry["active"][model_name] = version
    _write(registry_path, registry)
    return entry


def list_model_versions(
    registry_path: Path, model_name: str | None = None
) -> list[dict[str, Any]]:
    registry = _load(registry_path)
    models = registry["models"]
    if model_name is None:
        return models
    return [model for model in models if model.get("model_name") == model_name]


def get_model_version(
    registry_path: Path, model_name: str, version: str
) -> dict[str, Any] | None:
    for model in list_model_versions(registry_path, model_name):
        if model.get("version") == version:
            return model
    return None


def get_active_model(registry_path: Path, model_name: str) -> dict[str, Any] | None:
    registry = _load(registry_path)
    version = registry["active"].get(model_name)
    if version is None:
        return None
    return get_model_version(registry_path, model_name, str(version))


def set_active_model(
    registry_path: Path, model_name: str, version: str
) -> dict[str, Any]:
    model = get_model_version(registry_path, model_name, version)
    if model is None:
        raise ValueError(f"model version not found: {model_name}:{version}")
    registry = _load(registry_path)
    registry["active"][model_name] = version
    _write(registry_path, registry)
    return model
