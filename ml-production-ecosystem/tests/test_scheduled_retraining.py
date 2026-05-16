from pathlib import Path
import json

from ml_production_ecosystem.production_patterns.scheduled_retraining import run_scheduled_retraining


def test_scheduled_retraining_returns_completed_summary_and_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "scheduled-retraining.json"

    def fake_run_retraining(
        config_path: Path,
        set_active: bool,
        registry_path: Path | None = None,
        model_name: str = "movielens-popularity",
        require_quality_gate: bool = False,
    ) -> dict[str, object]:
        return {
            "status": "completed",
            "model_name": "recommendation",
            "version": "foundation-config-v1",
            "quality_gate": {"passed": True, "failures": []},
        }

    summary = run_scheduled_retraining(
        config_path=Path("configs/foundation-recommender.yaml"),
        set_active=True,
        require_quality_gate=True,
        output_path=output_path,
        retraining_runner=fake_run_retraining,
    )

    assert summary == {
        "status": "completed",
        "config_path": "configs/foundation-recommender.yaml",
        "set_active": True,
        "require_quality_gate": True,
        "run_id": "foundation-config-v1",
        "model_name": "recommendation",
        "quality_gate": {"status": "passed"},
    }
    assert json.loads(output_path.read_text()) == summary


def test_scheduled_retraining_returns_failed_summary_for_quality_gate_failure() -> None:
    def fake_run_retraining(
        config_path: Path,
        set_active: bool,
        registry_path: Path | None = None,
        model_name: str = "movielens-popularity",
        require_quality_gate: bool = False,
    ) -> dict[str, object]:
        return {
            "status": "failed_quality_gate",
            "model_name": "recommendation",
            "version": "foundation-config-v1",
            "quality_gate": {"passed": False, "failures": ["candidate_count 0.0 below minimum 1.0"]},
        }

    summary = run_scheduled_retraining(
        config_path=Path("configs/foundation-recommender.yaml"),
        set_active=True,
        require_quality_gate=True,
        retraining_runner=fake_run_retraining,
    )

    assert summary == {
        "status": "failed",
        "config_path": "configs/foundation-recommender.yaml",
        "set_active": True,
        "require_quality_gate": True,
        "run_id": "foundation-config-v1",
        "model_name": "recommendation",
        "quality_gate": {
            "status": "failed",
            "failures": ["candidate_count 0.0 below minimum 1.0"],
        },
        "error": "quality gate failed",
    }


def test_scheduled_retraining_returns_failed_summary_for_exception() -> None:
    def fake_run_retraining(
        config_path: Path,
        set_active: bool,
        registry_path: Path | None = None,
        model_name: str = "movielens-popularity",
        require_quality_gate: bool = False,
    ) -> dict[str, object]:
        raise RuntimeError("training failed")

    summary = run_scheduled_retraining(
        config_path=Path("configs/foundation-recommender.yaml"),
        set_active=False,
        require_quality_gate=False,
        retraining_runner=fake_run_retraining,
    )

    assert summary == {
        "status": "failed",
        "config_path": "configs/foundation-recommender.yaml",
        "set_active": False,
        "require_quality_gate": False,
        "error": "training failed",
    }
