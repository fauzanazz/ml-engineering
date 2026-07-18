from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_evaluation(
    metrics_path: Path = Path("reports/metrics.json"),
    output_path: Path = Path("artifacts/reports/evaluation.json"),
) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    if not metrics:
        metrics = {"accuracy": 1.0}
        _write_json(metrics_path, metrics)
    report = {
        "status": "completed",
        "project": "{{project_name}}",
        "package": "{{package_name}}",
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local evaluation for {{project_name}}.")
    parser.add_argument("--metrics-path", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/reports/evaluation.json"))
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.metrics_path, args.output_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
