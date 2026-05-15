from pathlib import Path
import json

from production_patterns.continual_learning import build_continual_learning_decision


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
    assert decision["approved_for_retraining"] is True


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
