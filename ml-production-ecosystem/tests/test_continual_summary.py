"""Tests for continual-learning history summarization."""

from pathlib import Path
import json
import subprocess
import sys

from production_patterns.continual_summary import summarize_continual_history


def _write_history(history_path: Path, entries: list[dict]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w") as file:
        for entry in entries:
            file.write(json.dumps(entry) + "\n")


def test_summary_empty_when_history_missing(tmp_path: Path) -> None:
    summary = summarize_continual_history(
        tmp_path / "missing.jsonl", tmp_path / "summary.json"
    )
    assert summary["status"] == "empty"
    assert summary["total_decisions"] == 0
    assert summary["latest_decision"] is None


def test_summary_marks_recurring_retrain(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    _write_history(
        history,
        [
            {"action": "retrain", "trigger": "drift", "approved_for_retraining": True},
            {"action": "monitor", "trigger": "scheduled-check", "approved_for_retraining": False},
            {"action": "retrain", "trigger": "drift", "approved_for_retraining": True},
        ],
    )

    summary = summarize_continual_history(history, tmp_path / "summary.json")

    assert summary["status"] == "recurring-retrain"
    assert summary["total_decisions"] == 3
    assert summary["action_counts"]["retrain"] == 2
    assert summary["trigger_counts"]["drift"] == 2
    assert summary["approved_for_retraining"] == 2
    assert summary["latest_decision"]["action"] == "retrain"


def test_summary_marks_investigation(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    _write_history(
        history,
        [
            {"action": "monitor", "trigger": "scheduled-check"},
            {"action": "investigate", "trigger": "deployment-demo"},
        ],
    )
    summary = summarize_continual_history(history, tmp_path / "summary.json")
    assert summary["status"] == "needs-investigation"


def test_summary_marks_stable(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    _write_history(
        history,
        [
            {"action": "monitor", "trigger": "scheduled-check"},
            {"action": "monitor", "trigger": "scheduled-check"},
        ],
    )
    summary = summarize_continual_history(history, tmp_path / "summary.json")
    assert summary["status"] == "stable"


def test_summary_cli_writes_output(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    _write_history(
        history,
        [{"action": "retrain", "trigger": "drift", "approved_for_retraining": True}],
    )
    output = tmp_path / "out.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "production_patterns.continual_summary",
            "--history-path",
            str(history),
            "--output-path",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(output.read_text())
    assert payload["total_decisions"] == 1
    assert "status" in payload
    assert "retrain" in result.stdout
