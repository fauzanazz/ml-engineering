"""Simple metric-threshold quality gate for production-pattern retraining."""

from pathlib import Path
import json
from typing import Any


def _resolve_metrics_path(config: dict[str, Any] | None, base: Path | None = None) -> Path | None:
    if not config:
        return None
    metrics_path = config.get("metrics_path")
    if metrics_path is None:
        return None

    path = Path(str(metrics_path))
    if path.is_absolute() or base is None:
        return path

    candidate_paths: list[Path] = [base / path]
    if base.name == "configs":
        candidate_paths.append(base.parent / path)
    candidate_paths.append(Path.cwd() / path)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    return candidate_paths[0]


def evaluate_quality_gate(config: dict[str, Any] | None, base: Path | None = None) -> dict[str, object]:
    if not config or not bool(config.get("enabled", False)):
        return {"passed": True, "failures": []}

    metrics_path = _resolve_metrics_path(config, base=base)
    if metrics_path is None:
        return {"passed": False, "failures": ["quality_gate.metrics_path is required"]}

    with metrics_path.open() as file:
        metrics = json.load(file)

    failures = []
    minimums = config.get("minimums", {})
    if not isinstance(minimums, dict):
        return {"passed": False, "failures": ["quality_gate.minimums must be a mapping"]}

    for metric_name, minimum_value in minimums.items():
        actual_value = float(metrics.get(metric_name, 0.0))
        threshold = float(minimum_value)
        if actual_value < threshold:
            failures.append(f"{metric_name} {actual_value} below minimum {threshold}")

    return {"passed": not failures, "failures": failures}
