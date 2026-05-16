"""Scheduler-friendly retraining entrypoint with JSON summary output."""

import argparse
import json
from pathlib import Path
from typing import Callable

from .retraining import run_retraining

RetrainingRunner = Callable[[Path, bool, Path | None, str, bool], dict[str, object]]
DEFAULT_MODEL_NAME = "movielens-popularity"


def _quality_gate_summary(result: dict[str, object]) -> dict[str, object]:
    quality_gate = result.get("quality_gate", {})
    if not isinstance(quality_gate, dict):
        return {"status": "passed"}

    if bool(quality_gate.get("passed", True)):
        return {"status": "passed"}

    failures = quality_gate.get("failures", [])
    return {"status": "failed", "failures": failures}


def _write_report(output_path: Path | None, summary: dict[str, object]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


def run_scheduled_retraining(
    config_path: Path,
    set_active: bool = False,
    require_quality_gate: bool = False,
    output_path: Path | None = None,
    registry_path: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    retraining_runner: RetrainingRunner = run_retraining,
) -> dict[str, object]:
    try:
        result = retraining_runner(config_path, set_active, registry_path, model_name, require_quality_gate)
    except Exception as error:
        summary: dict[str, object] = {
            "status": "failed",
            "config_path": str(config_path),
            "set_active": set_active,
            "require_quality_gate": require_quality_gate,
            "error": str(error),
        }
        _write_report(output_path, summary)
        return summary

    quality_gate = _quality_gate_summary(result)
    status = "completed" if result.get("status") != "failed_quality_gate" else "failed"
    summary = {
        "status": status,
        "config_path": str(config_path),
        "set_active": set_active,
        "require_quality_gate": require_quality_gate,
        "run_id": str(result.get("version")),
        "model_name": str(result.get("model_name")),
        "quality_gate": quality_gate,
    }
    if status == "failed":
        summary["error"] = "quality gate failed"

    _write_report(output_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scheduler-friendly production retraining.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set-active", action="store_true")
    parser.add_argument("--require-quality-gate", action="store_true")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    summary = run_scheduled_retraining(
        config_path=args.config,
        set_active=args.set_active,
        require_quality_gate=args.require_quality_gate,
        output_path=args.output_path,
        registry_path=args.registry_path,
        model_name=args.model_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
