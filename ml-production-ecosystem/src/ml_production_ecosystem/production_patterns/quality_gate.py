"""Simple metric-threshold quality gate for production-pattern retraining."""

from pathlib import Path
import json
from typing import Any


def _resolve_path(value: object, base: Path | None = None) -> Path | None:
    if value is None:
        return None

    path = Path(str(value))
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


def _resolve_metrics_path(config: dict[str, Any] | None, base: Path | None = None) -> Path | None:
    if not config:
        return None
    return _resolve_path(config.get("metrics_path"), base=base)


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open() as file:
        metrics = json.load(file)
    return metrics if isinstance(metrics, dict) else {}


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any] | str:
    value = config.get(key, {})
    if not isinstance(value, dict):
        return f"quality_gate.{key} must be a mapping"
    return value


def evaluate_quality_gate(config: dict[str, Any] | None, base: Path | None = None) -> dict[str, object]:
    if not config or not bool(config.get("enabled", False)):
        return {"passed": True, "failures": []}

    metrics_path = _resolve_metrics_path(config, base=base)
    if metrics_path is None:
        return {"passed": False, "failures": ["quality_gate.metrics_path is required"]}

    metrics = _load_metrics(metrics_path)
    failures: list[str] = []
    minimums = _require_mapping(config, "minimums")
    minimum_deltas = _require_mapping(config, "minimum_deltas")
    maximum_regressions = _require_mapping(config, "maximum_regressions")
    for mapping in (minimums, minimum_deltas, maximum_regressions):
        if isinstance(mapping, str):
            return {"passed": False, "failures": [mapping]}

    for metric_name, minimum_value in minimums.items():
        actual_value = float(metrics.get(metric_name, 0.0))
        threshold = float(minimum_value)
        if actual_value < threshold:
            failures.append(f"{metric_name} {actual_value} below minimum {threshold}")

    baseline_path = _resolve_path(
        config.get("baseline_metrics_path") or config.get("champion_metrics_path"),
        base=base,
    )
    if baseline_path is None:
        if bool(config.get("baseline_required", False)):
            failures.append("quality_gate.baseline_metrics_path is required")
        return {"passed": not failures, "failures": failures}
    if not baseline_path.exists():
        if bool(config.get("baseline_required", False)):
            failures.append(f"quality_gate.baseline_metrics_path not found: {baseline_path}")
        return {"passed": not failures, "failures": failures}

    baseline_metrics = _load_metrics(baseline_path)
    for metric_name, minimum_delta in minimum_deltas.items():
        actual_delta = float(metrics.get(metric_name, 0.0)) - float(baseline_metrics.get(metric_name, 0.0))
        threshold = float(minimum_delta)
        if actual_delta < threshold:
            failures.append(f"{metric_name} delta {actual_delta} below minimum {threshold}")

    for metric_name, maximum_regression in maximum_regressions.items():
        regression = float(baseline_metrics.get(metric_name, 0.0)) - float(metrics.get(metric_name, 0.0))
        threshold = float(maximum_regression)
        if regression > threshold:
            failures.append(f"{metric_name} regression {regression} above maximum {threshold}")

    return {"passed": not failures, "failures": failures}
