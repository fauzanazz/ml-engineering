from pathlib import Path
import json

from ml_production_ecosystem.production_patterns.continual_learning import build_continual_learning_decision


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_continual_learning_monitors_without_trigger(tmp_path: Path) -> None:
    drift_path = _write_json(tmp_path / "drift.json", {"status": "passed"})
    demo_path = _write_json(tmp_path / "demo.json", {"status": "passed"})

    decision = build_continual_learning_decision(drift_path, demo_path, tmp_path / "decision.json")

    assert decision["action"] == "monitor"
    assert decision["approved_for_retraining"] is False


def test_continual_learning_retrains_on_drift_breach(tmp_path: Path) -> None:
    drift_path = _write_json(tmp_path / "drift.json", {"status": "failed"})
    demo_path = _write_json(tmp_path / "demo.json", {"status": "passed"})

    decision = build_continual_learning_decision(drift_path, demo_path, tmp_path / "decision.json")

    assert decision["action"] == "retrain"
    assert decision["trigger"] == "drift"
    assert decision["approved_for_retraining"] is False


def test_continual_learning_investigates_failed_deployment_demo(tmp_path: Path) -> None:
    drift_path = _write_json(tmp_path / "drift.json", {"status": "failed"})
    demo_path = _write_json(tmp_path / "demo.json", {"status": "failed"})
    output_path = tmp_path / "decision.json"

    decision = build_continual_learning_decision(drift_path, demo_path, output_path)

    assert decision["action"] == "investigate"
    assert decision["trigger"] == "deployment-demo"
    assert decision["approved_for_retraining"] is False
    assert json.loads(output_path.read_text()) == decision

def test_continual_learning_appends_history_for_monitoring(tmp_path: Path) -> None:
    drift_path = _write_json(tmp_path / "drift.json", {"status": "passed"})
    demo_path = _write_json(tmp_path / "demo.json", {"status": "passed"})
    history_path = tmp_path / "history.jsonl"

    first = build_continual_learning_decision(
        drift_path,
        demo_path,
        tmp_path / "decision-1.json",
        history_path,
    )
    second = build_continual_learning_decision(
        drift_path,
        demo_path,
        tmp_path / "decision-2.json",
        history_path,
    )

    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert history == [first, second]
    assert all(row["action"] == "monitor" for row in history)

def test_continual_learning_drift_recommendation_omits_set_active(tmp_path: Path) -> None:
    drift_path = _write_json(tmp_path / "drift.json", {"status": "failed"})
    demo_path = _write_json(tmp_path / "demo.json", {"status": "passed"})

    decision = build_continual_learning_decision(drift_path, demo_path, tmp_path / "decision.json")

    recommendation = decision["recommendation"]
    command = recommendation["command"]
    assert command[:3] == ["uv", "run", "production-scheduled-retrain"]
    assert "--set-active" not in command
    assert "--require-quality-gate" in command
    assert "--config" in command
    assert "configs/foundation-recommender.yaml" in command
    assert recommendation["approval_required"] is True


def test_continual_learning_drift_recommendation_uses_custom_retraining_config_path(tmp_path: Path) -> None:
    drift_path = _write_json(tmp_path / "drift.json", {"status": "failed"})
    demo_path = _write_json(tmp_path / "demo.json", {"status": "passed"})

    decision = build_continual_learning_decision(
        drift_path,
        demo_path,
        tmp_path / "decision.json",
        retraining_config_path=Path("configs/custom-pipeline.yaml"),
    )

    assert "configs/custom-pipeline.yaml" in decision["recommendation"]["command"]


def test_continual_learning_monitor_and_investigate_actions_have_no_recommendation(tmp_path: Path) -> None:
    monitor_drift = _write_json(tmp_path / "drift-ok.json", {"status": "passed"})
    monitor_demo = _write_json(tmp_path / "demo-ok.json", {"status": "passed"})
    monitor_decision = build_continual_learning_decision(monitor_drift, monitor_demo, tmp_path / "monitor.json")
    assert "recommendation" not in monitor_decision

    investigate_drift = _write_json(tmp_path / "drift-fail.json", {"status": "failed"})
    investigate_demo = _write_json(tmp_path / "demo-fail.json", {"status": "failed"})
    investigate_decision = build_continual_learning_decision(
        investigate_drift, investigate_demo, tmp_path / "investigate.json"
    )
    assert "recommendation" not in investigate_decision
