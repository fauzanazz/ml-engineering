from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKLIST_PATH = ROOT / "docs" / "domains" / "production-patterns" / "release-checklist.md"


def test_release_checklist_has_required_sections() -> None:
    assert CHECKLIST_PATH.exists()
    checklist = CHECKLIST_PATH.read_text()

    for section in [
        "Pre-release",
        "Train Candidate",
        "Quality Gate",
        "Activate Model",
        "Serving Verification",
        "Monitoring And Alerts",
        "Rollback Plan",
        "Post-release Notes",
    ]:
        assert f"## {section}" in checklist


def test_release_checklist_references_release_flow_steps() -> None:
    checklist = CHECKLIST_PATH.read_text()

    assert "production-scheduled-retrain" in checklist
    assert "production-monitor" in checklist
    assert "alerting-runbook.md" in checklist
    assert "production-rollback-model" in checklist
    assert "production-canary-decision" in checklist
    assert "scheduled_retraining.py" in checklist
    assert "monitoring_loop.py" in checklist
    assert "alerts/rules.yaml" in checklist
    assert "rollback.py" in checklist


def test_release_checklist_contains_copy_paste_commands() -> None:
    checklist = CHECKLIST_PATH.read_text()

    assert "uv run production-scheduled-retrain \\" in checklist
    assert "--config configs/foundation-recommender.yaml \\" in checklist
    assert "--set-active \\" in checklist
    assert "--require-quality-gate \\" in checklist
    assert "--output-path artifacts/reports/production-patterns/scheduled-retraining.json" in checklist
    assert "uv run production-monitor \\" in checklist
    assert "--base-url http://127.0.0.1:8000 \\" in checklist
    assert "--max-error-count 0 \\" in checklist
    assert "--max-drift-score 0.2 \\" in checklist
    assert "--max-latency-ms-last 100" in checklist
    assert "uv run production-rollback-model \\" in checklist
    assert "--registry-path registry/models.json \\" in checklist
    assert "--model-name movielens-popularity \\" in checklist
    assert "--target-version foundation-config-v1 \\" in checklist
    assert '--reason "release verification failed"' in checklist
