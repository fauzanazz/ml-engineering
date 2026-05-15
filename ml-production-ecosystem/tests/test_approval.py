from pathlib import Path
import json

from production_patterns.approval import build_approval_decision


def _write_validation_report(tmp_path: Path, status: str) -> Path:
    path = tmp_path / "offline-validation.json"
    path.write_text(
        json.dumps(
            {
                "status": status,
                "model": {"name": "demo", "version": "v1", "type": "classifier"},
                "quality_gate": {"passed": status == "passed", "failures": []},
            }
        )
    )
    return path


def test_build_approval_decision_pending_without_manual_approval(tmp_path: Path) -> None:
    output_path = tmp_path / "approval.json"

    decision = build_approval_decision(_write_validation_report(tmp_path, "passed"), output_path=output_path)

    assert decision["status"] == "pending"
    assert decision["approved"] is False
    assert decision["checks"]["offline_validation_passed"] is True
    assert decision["checks"]["manual_approval_provided"] is False
    assert json.loads(output_path.read_text()) == decision


def test_build_approval_decision_approves_passed_validation(tmp_path: Path) -> None:
    decision = build_approval_decision(
        _write_validation_report(tmp_path, "passed"),
        approved=True,
        approver="tester",
        output_path=tmp_path / "approval.json",
    )

    assert decision["status"] == "approved"
    assert decision["approved"] is True
    assert decision["approver"] == "tester"


def test_build_approval_decision_rejects_failed_validation(tmp_path: Path) -> None:
    decision = build_approval_decision(
        _write_validation_report(tmp_path, "failed"),
        approved=True,
        output_path=tmp_path / "approval.json",
    )

    assert decision["status"] == "rejected"
    assert decision["approved"] is False
    assert decision["reason"] == "offline validation failed"
