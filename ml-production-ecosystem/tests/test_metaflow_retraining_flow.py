from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
FLOW_PATH = ROOT / "configs" / "production-patterns" / "metaflow" / "retraining_flow.py"


def test_retraining_flow_skeleton_exists_with_expected_steps_and_commands() -> None:
    assert FLOW_PATH.exists()
    source = FLOW_PATH.read_text()

    assert "uv run production-scheduled-retrain" not in source  # tuple form, not shell string
    assert '"production-scheduled-retrain"' in source
    assert "--config configs/foundation-recommender.yaml" not in source
    assert '"configs/foundation-recommender.yaml"' in source
    assert '"--set-active"' in source
    assert '"--require-quality-gate"' in source
    assert '"artifacts/reports/production-patterns/scheduled-retraining.json"' in source
    assert '"production-monitor"' in source
    assert '"http://127.0.0.1:8000"' in source
    assert '"--max-error-count"' in source and '"0"' in source
    assert '"--max-drift-score"' in source and '"0.2"' in source
    assert '"--max-latency-ms-last"' in source and '"100"' in source

    # Step order is declared through the self.next(...) chain, mirroring the
    # Airflow DAG's `validate_config >> scheduled_retrain >> monitor_after_retrain`.
    start_idx = source.index("def start(")
    to_scheduled_idx = source.index("self.next(self.scheduled_retrain)")
    scheduled_idx = source.index("def scheduled_retrain(")
    to_monitor_idx = source.index("self.next(self.monitor_after_retrain)")
    monitor_idx = source.index("def monitor_after_retrain(")
    to_end_idx = source.index("self.next(self.end)")
    end_idx = source.index("def end(")
    assert start_idx < to_scheduled_idx < scheduled_idx < to_monitor_idx < monitor_idx < to_end_idx < end_idx


def test_retraining_flow_imports_without_metaflow_installed_and_runs_steps_in_order() -> None:
    spec = importlib.util.spec_from_file_location("retraining_flow", FLOW_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: list[tuple[str, ...]] = []
    module.run_command = calls.append

    flow = module.ProductionRetrainingFlow()
    flow.start()
    flow.scheduled_retrain()
    flow.monitor_after_retrain()

    assert calls == [
        module.VALIDATE_CONFIG_COMMAND,
        module.SCHEDULED_RETRAIN_COMMAND,
        module.MONITOR_AFTER_RETRAIN_COMMAND,
    ]
    assert module.SCHEDULED_RETRAIN_COMMAND[:3] == ("uv", "run", "production-scheduled-retrain")
    assert "--set-active" in module.SCHEDULED_RETRAIN_COMMAND
    assert module.MONITOR_AFTER_RETRAIN_COMMAND[:3] == ("uv", "run", "production-monitor")
