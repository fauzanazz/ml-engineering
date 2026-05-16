from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
DAG_PATH = ROOT / "configs" / "production-patterns" / "airflow" / "retraining_dag.py"


def test_retraining_dag_skeleton_exists_with_expected_tasks_and_commands() -> None:
    assert DAG_PATH.exists()
    source = DAG_PATH.read_text()

    assert "validate_config" in source
    assert "scheduled_retrain" in source
    assert "monitor_after_retrain" in source
    assert "validate_config >> scheduled_retrain >> monitor_after_retrain" in source
    assert "uv run production-scheduled-retrain" in source
    assert "--config configs/foundation-recommender.yaml" in source
    assert "--set-active" in source
    assert "--require-quality-gate" in source
    assert "--output-path artifacts/reports/production-patterns/scheduled-retraining.json" in source
    assert "uv run production-monitor" in source
    assert "--base-url http://127.0.0.1:8000" in source
    assert "--max-error-count 0" in source
    assert "--max-drift-score 0.2" in source
    assert "--max-latency-ms-last 100" in source


def test_retraining_dag_imports_without_airflow_installed() -> None:
    spec = importlib.util.spec_from_file_location("retraining_dag", DAG_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.dag.dag_id == "production_retraining"
    assert [task.task_id for task in module.dag.tasks] == [
        "validate_config",
        "scheduled_retrain",
        "monitor_after_retrain",
    ]
