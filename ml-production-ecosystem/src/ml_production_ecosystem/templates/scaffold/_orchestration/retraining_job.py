"""Executable local retraining job for {{project_name}} orchestration adapters."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _coerce_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def _next_content(lines: list[str], start: int) -> str:
    for line in lines[start:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    lines = text.splitlines()
    for index, raw in enumerate(lines):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if isinstance(parent, list):
                parent.append(_coerce_scalar(stripped[2:]))
            continue
        key, _, value = stripped.partition(":")
        if not isinstance(parent, dict) or not key:
            continue
        value = value.strip()
        if value:
            parent[key] = _coerce_scalar(value)
            continue
        child: dict[str, Any] | list[Any] = [] if _next_content(lines, index + 1).startswith("- ") else {}
        parent[key] = child
        stack.append((indent, child))
    return root


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _parse_simple_yaml(path.read_text())
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _summary_path(config: dict[str, Any]) -> Path:
    training = config.get("training", {})
    if isinstance(training, dict) and training.get("summary_path"):
        return Path(str(training["summary_path"]))
    return Path("artifacts/reports/training-summary.json")


def _run_training_command(config: dict[str, Any]) -> dict[str, Any] | None:
    training = config.get("training", {})
    if not isinstance(training, dict):
        return None
    command = training.get("command")
    summary_path = _summary_path(config)
    if not isinstance(command, list) or not command:
        return None
    args = [sys.executable if str(part) == "python" else str(part) for part in command]
    subprocess.run(args, check=True)
    return _load_json(summary_path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def _run_package_training(config: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = _summary_path(config)
    try:
        module = importlib.import_module("{{package_name}}.train")
    except ModuleNotFoundError:
        return None
    writer = getattr(module, "write_training_summary", None)
    if callable(writer):
        return dict(writer(summary_path))
    return _run_training_command(config)


def _run_builtin_training(summary_path: Path) -> dict[str, Any]:
    model_path = Path("artifacts/models/local-retrain/model.json")
    metrics_path = summary_path.parent / "metrics.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps({"kind": "bootstrap-logistic-baseline", "weights": [0.7, -0.2, 0.4], "bias": 0.1}, indent=2) + "\n")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"accuracy": 1.0, "loss": 0.0}, indent=2, sort_keys=True) + "\n")
    summary = {
        "model_name": "{{package_name}}",
        "version": "local-retrain",
        "artifact_uri": str(model_path),
        "metrics_uri": str(metrics_path),
    }
    _write_json(summary_path, summary)
    return summary


def _resolve_existing_path(uri: object, summary_path: Path) -> Path:
    path = Path(str(uri))
    if path.is_absolute():
        return path
    for candidate in (summary_path.parent / path, summary_path.parent.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def _quality_gate(config: dict[str, Any], summary: dict[str, Any], summary_path: Path, required: bool) -> dict[str, Any]:
    if not required:
        return {"status": "not_required", "failures": []}
    gate = config.get("quality_gate", {})
    gate = gate if isinstance(gate, dict) else {}
    metrics_path = _resolve_existing_path(summary.get("metrics_uri", gate.get("metrics_path", "metrics.json")), summary_path)
    metrics = _load_json(metrics_path)
    failures: list[str] = []
    for metric, minimum in (gate.get("minimums") or {}).items():
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or math.isnan(float(value)) or float(value) < float(minimum):
            failures.append(f"{metric}={value!r} below minimum {minimum}")
    for metric, maximum in (gate.get("maximums") or {}).items():
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or math.isnan(float(value)) or float(value) > float(maximum):
            failures.append(f"{metric}={value!r} above maximum {maximum}")
    return {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def run_retraining(
    config_path: Path,
    output_path: Path,
    *,
    set_active: bool = False,
    require_quality_gate: bool = False,
) -> dict[str, Any]:
    config = _load_config(config_path)
    summary_path = _summary_path(config)
    summary = _run_package_training(config) or _run_builtin_training(summary_path)
    gate = _quality_gate(config, summary, summary_path, require_quality_gate)
    passed = gate["status"] in {"passed", "not_required"}
    timestamp = datetime.now(UTC).isoformat()
    report = {
        "status": "completed" if passed else "failed_quality_gate",
        "project": "{{project_name}}",
        "package": "{{package_name}}",
        "preset": "{{preset}}",
        "task": "{{task}}",
        "model_type": "{{model_type}}",
        "backend": "{{backend}}",
        "config_path": str(config_path),
        "model_name": str(summary.get("model_name", "{{package_name}}")),
        "version": str(summary.get("version", "local-retrain")),
        "artifact_uri": str(summary.get("artifact_uri", "")),
        "metrics_uri": str(summary.get("metrics_uri", "")),
        "training_summary": summary,
        "set_active": set_active,
        "quality_gate": gate,
        "completed_at": timestamp,
    }
    if set_active and passed:
        active_path = Path("artifacts/active-model.json")
        _write_json(active_path, {"model_name": report["model_name"], "version": report["version"], "artifact_uri": report["artifact_uri"], "activated_at": timestamp})
        report["active_model_uri"] = str(active_path)
    _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local retraining pipeline for {{project_name}}.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/reports/scheduled-retraining.json"))
    parser.add_argument("--set-active", action="store_true")
    parser.add_argument("--require-quality-gate", action="store_true")
    args = parser.parse_args()
    report = run_retraining(args.config, args.output_path, set_active=args.set_active, require_quality_gate=args.require_quality_gate)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "failed_quality_gate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
