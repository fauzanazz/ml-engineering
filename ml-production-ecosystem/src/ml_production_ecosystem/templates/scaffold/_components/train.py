from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_training_summary(summary_path: Path) -> dict[str, str]:
    summary = {
        "model_name": "{{package_name}}",
        "version": "local-dev",
        "artifact_uri": "models/local-dev",
        "metrics_uri": "reports/metrics.json",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    metrics_path = summary_path.parent.parent / "reports" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"accuracy": 1.0}, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local training for {{project_name}}.")
    parser.add_argument("--summary-path", type=Path, default=Path("artifacts/reports/training-summary.json"))
    args = parser.parse_args()
    print(json.dumps(write_training_summary(args.summary_path), sort_keys=True))


if __name__ == "__main__":
    main()
