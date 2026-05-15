"""Build release evidence summary from production pattern reports."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be object: {path}")
    return data


def _write_summary(output_path: Path | None, summary: dict[str, object]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


def _blocked_error(message: str, output_path: Path | None) -> dict[str, object]:
    summary: dict[str, object] = {"status": "blocked", "error": message}
    _write_summary(output_path, summary)
    return summary


def build_release_summary(
    retraining_report_path: Path,
    deployment_manifest_path: Path,
    monitor_summary_path: Path,
    smoke_status: str,
    rollback_target: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    for input_path in [retraining_report_path, deployment_manifest_path, monitor_summary_path]:
        if not input_path.exists():
            return _blocked_error(f"missing input file: {input_path}", output_path)

    try:
        retraining_report = _read_json(retraining_report_path)
        deployment_manifest = _read_yaml(deployment_manifest_path)
        monitor_summary = _read_json(monitor_summary_path)
    except Exception as error:
        return _blocked_error(str(error), output_path)

    quality_gate = retraining_report.get("quality_gate", {})
    if not isinstance(quality_gate, dict):
        quality_gate = {"status": "unknown"}

    quality_gate_status = quality_gate.get("status")
    monitor_status = str(monitor_summary.get("status"))
    status = "ready" if quality_gate_status == "passed" and smoke_status == "passed" and monitor_status == "healthy" else "blocked"

    summary = {
        "status": status,
        "model_name": str(retraining_report.get("model_name")),
        "run_id": str(retraining_report.get("run_id")),
        "quality_gate": quality_gate,
        "service_name": str(deployment_manifest.get("service_name")),
        "image": str(deployment_manifest.get("image")),
        "smoke_status": smoke_status,
        "monitor_status": monitor_status,
        "rollback_target": rollback_target,
    }
    _write_summary(output_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build release summary evidence report.")
    parser.add_argument("--retraining-report", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--monitor-summary", type=Path, required=True)
    parser.add_argument("--smoke-status", required=True)
    parser.add_argument("--rollback-target", required=True)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    summary = build_release_summary(
        retraining_report_path=args.retraining_report,
        deployment_manifest_path=args.deployment_manifest,
        monitor_summary_path=args.monitor_summary,
        smoke_status=args.smoke_status,
        rollback_target=args.rollback_target,
        output_path=args.output_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
