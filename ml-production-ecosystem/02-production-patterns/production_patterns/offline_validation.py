"""Generic offline validation report for model candidates."""

from pathlib import Path
import argparse
import json
from typing import Any

import yaml

from .quality_gate import evaluate_quality_gate

DEFAULT_OUTPUT_PATH = Path("02-production-patterns/reports/offline-validation.json")


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as file:
        config = yaml.safe_load(file)
    if isinstance(config, dict):
        return config
    return {}


def _quality_gate_config(config: dict[str, Any]) -> dict[str, Any]:
    quality_gate = config.get("quality_gate", {})
    if isinstance(quality_gate, dict):
        return quality_gate
    return {}


def _load_metrics(quality_gate: dict[str, Any]) -> dict[str, float]:
    metrics_path = quality_gate.get("metrics_path")
    if metrics_path is None or not Path(str(metrics_path)).exists():
        return {}
    metrics = json.loads(Path(str(metrics_path)).read_text())
    return {key: float(value) for key, value in metrics.items() if isinstance(value, int | float)}


def _model_identity(config: dict[str, Any]) -> dict[str, str | None]:
    pipeline = config.get("pipeline", {})
    model = config.get("model", {})
    return {
        "name": str(pipeline.get("name")) if isinstance(pipeline, dict) and pipeline.get("name") else None,
        "version": str(pipeline.get("version")) if isinstance(pipeline, dict) and pipeline.get("version") else None,
        "type": str(model.get("type")) if isinstance(model, dict) and model.get("type") else None,
    }


def build_offline_validation_report(config_path: Path, output_path: Path = DEFAULT_OUTPUT_PATH) -> dict[str, object]:
    config = _load_config(config_path)
    quality_gate = _quality_gate_config(config)
    result = evaluate_quality_gate(quality_gate)
    report = {
        "status": "passed" if result["passed"] else "failed",
        "model": _model_identity(config),
        "quality_gate": result,
        "metrics": _load_metrics(quality_gate),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline validation report from model metrics and quality gate.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_offline_validation_report(args.config, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
