"""Continual-learning decision from local monitoring artifacts."""

from pathlib import Path
import argparse
import json
from typing import Any

DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/continual-learning-decision.json")
DEFAULT_HISTORY_PATH = Path("artifacts/reports/production-patterns/continual-learning-history.jsonl")


def _read_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def build_continual_learning_decision(
    drift_report_path: Path | None = None,
    deployment_demo_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    history_path: Path | None = None,
) -> dict[str, object]:
    drift = _read_report(drift_report_path)
    demo = _read_report(deployment_demo_path)

    if demo.get("status") == "failed":
        decision = {
            "action": "investigate",
            "trigger": "deployment-demo",
            "approved_for_retraining": False,
            "reason": "deployment demo failed",
        }
    elif drift.get("status") == "failed":
        decision = {
            "action": "retrain",
            "trigger": "drift",
            "approved_for_retraining": True,
            "reason": "drift threshold breached",
        }
    else:
        decision = {
            "action": "monitor",
            "trigger": "scheduled-check",
            "approved_for_retraining": False,
            "reason": "no retraining trigger",
        }

    decision["evidence"] = {
        "drift_report_path": str(drift_report_path) if drift_report_path else None,
        "deployment_demo_path": str(deployment_demo_path) if deployment_demo_path else None,
        "drift_status": drift.get("status"),
        "deployment_demo_status": demo.get("status"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a") as history_file:
            history_file.write(json.dumps(decision, sort_keys=True) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Write continual-learning decision from local reports.")
    parser.add_argument("--drift-report", type=Path)
    parser.add_argument("--deployment-demo", type=Path)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    args = parser.parse_args()

    decision = build_continual_learning_decision(
        args.drift_report,
        args.deployment_demo,
        args.output_path,
        args.history_path,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
