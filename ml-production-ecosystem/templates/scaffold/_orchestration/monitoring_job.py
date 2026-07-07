"""Executable local monitoring job for {{project_name}} orchestration adapters."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"input report not found: {path}")
    with path.open() as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError(f"input report must be a JSON object: {path}")
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_monitoring(input_path: Path, output_path: Path) -> dict[str, Any]:
    evidence = _load_report(input_path)
    quality_gate = evidence.get("quality_gate")
    checks = [
        {
            "name": "evidence_status",
            "passed": evidence.get("status") == "completed",
            "message": f"evidence status is {evidence.get('status')!r}",
        },
        {
            "name": "quality_gate",
            "passed": not isinstance(quality_gate, dict) or quality_gate.get("status") in {"passed", "not_required"},
            "message": f"quality gate is {quality_gate.get('status') if isinstance(quality_gate, dict) else 'not_present'!r}",
        },
    ]
    report = {
        "status": "healthy" if all(check["passed"] for check in checks) else "unhealthy",
        "project": "{{project_name}}",
        "package": "{{package_name}}",
        "backend": "{{backend}}",
        "input_path": str(input_path),
        "checks": checks,
        "checked_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local post-retraining monitoring evidence for {{project_name}}.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/reports/monitoring.json"))
    args = parser.parse_args()
    print(json.dumps(run_monitoring(args.input_path, args.output_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
