"""Local canary release decision from generic lifecycle evidence."""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = Path("02-production-patterns/reports/canary-decision.json")


def build_canary_decision(
    deployment_demo_path: Path,
    drift_report_path: Path,
    approval_path: Path | None = None,
    canary_percent: int = 10,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    failures = []
    if canary_percent <= 0 or canary_percent > 100:
        failures.append("canary_percent must be between 1 and 100")

    demo = _read_report(deployment_demo_path)
    drift = _read_report(drift_report_path)
    approval = _read_report(approval_path) if approval_path else {}

    demo_status = str(demo.get("status", "missing"))
    drift_status = str(drift.get("status", "missing"))
    approval_status = str(approval.get("status", "not-required"))

    if demo_status != "passed":
        failures.append(f"deployment demo status is {demo_status}")
    if drift_status != "passed":
        failures.append(f"drift status is {drift_status}")
    if approval_path is not None and approval_status != "approved":
        failures.append(f"approval status is {approval_status}")

    decision = "promote" if not failures else "rollback"
    report: dict[str, Any] = {
        "status": "passed" if not failures else "blocked",
        "decision": decision,
        "canary_percent": canary_percent,
        "evidence": {
            "deployment_demo_path": str(deployment_demo_path),
            "deployment_demo_status": demo_status,
            "drift_report_path": str(drift_report_path),
            "drift_status": drift_status,
            "approval_path": str(approval_path) if approval_path else None,
            "approval_status": approval_status,
        },
        "failures": failures,
        "rollback_required": decision == "rollback",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _read_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write local canary release decision from lifecycle reports.")
    parser.add_argument("--deployment-demo", type=Path, required=True)
    parser.add_argument("--drift-report", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--canary-percent", type=int, default=10)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_canary_decision(
        deployment_demo_path=args.deployment_demo,
        drift_report_path=args.drift_report,
        approval_path=args.approval,
        canary_percent=args.canary_percent,
        output_path=args.output_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
