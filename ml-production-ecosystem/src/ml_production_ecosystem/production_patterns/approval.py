"""Explicit model approval decision artifact for release workflows."""

from pathlib import Path
import argparse
from datetime import UTC, datetime
import json
from typing import Any

DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/approval-decision.json")


def _read_validation_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def build_approval_decision(
    validation_report_path: Path,
    approved: bool = False,
    approver: str = "local-operator",
    reason: str | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    validation = _read_validation_report(validation_report_path)
    validation_passed = validation.get("status") == "passed"
    decision_approved = bool(approved and validation_passed)
    if reason is None:
        if decision_approved:
            reason = "offline validation passed and approval flag provided"
        elif not validation_passed:
            reason = "offline validation failed"
        else:
            reason = "manual approval flag not provided"

    if decision_approved:
        status = "approved"
    elif validation_passed:
        status = "pending"
    else:
        status = "rejected"

    decision = {
        "status": status,
        "approved": decision_approved,
        "approver": approver,
        "reason": reason,
        "decided_at": datetime.now(UTC).isoformat(),
        "validation_report_path": str(validation_report_path),
        "model": validation.get("model", {}),
        "checks": {
            "offline_validation_passed": validation_passed,
            "manual_approval_provided": bool(approved),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicit model approval decision artifact.")
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--approver", default="local-operator")
    parser.add_argument("--reason")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    decision = build_approval_decision(
        validation_report_path=args.validation_report,
        approved=args.approve,
        approver=args.approver,
        reason=args.reason,
        output_path=args.output_path,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
