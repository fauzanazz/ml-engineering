from pathlib import Path
import json
import subprocess
import sys

from ml_production_ecosystem.recommendation.train import get_active_model, register_model_version
from ml_production_ecosystem.production_patterns.rollback import rollback_model


def _write_registry(registry_path: Path) -> None:
    register_model_version(
        registry_path=registry_path,
        model_name="movielens-popularity",
        version="bad-v2",
        artifact_uri="artifacts/recommendation/bad-v2",
        metrics_uri="artifacts/recommendation/bad-v2/metrics.json",
        set_active=True,
    )
    register_model_version(
        registry_path=registry_path,
        model_name="movielens-popularity",
        version="foundation-config-v1",
        artifact_uri="artifacts/recommendation/foundation-config-v1",
        metrics_uri="artifacts/recommendation/foundation-config-v1/metrics.json",
    )


def test_rollback_model_sets_target_version_active_and_writes_report(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    report_path = tmp_path / "reports" / "rollback.json"
    _write_registry(registry_path)

    summary = rollback_model(
        registry_path=registry_path,
        model_name="movielens-popularity",
        target_version="foundation-config-v1",
        reason="high drift after deploy",
        report_path=report_path,
    )

    assert summary == {
        "status": "rolled_back",
        "model_name": "movielens-popularity",
        "target_version": "foundation-config-v1",
        "reason": "high drift after deploy",
        "report_path": str(report_path),
    }
    active = get_active_model(registry_path, "movielens-popularity")
    assert active is not None
    assert active["version"] == "foundation-config-v1"
    assert json.loads(report_path.read_text()) == summary


def test_rollback_model_returns_failed_when_target_version_missing(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    _write_registry(registry_path)

    summary = rollback_model(
        registry_path=registry_path,
        model_name="movielens-popularity",
        target_version="missing-version",
        reason="high drift after deploy",
        report_path=tmp_path / "rollback.json",
    )

    assert summary == {
        "status": "failed",
        "model_name": "movielens-popularity",
        "target_version": "missing-version",
        "error": "target version not found",
    }


def test_rollback_model_returns_failed_when_model_missing(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    _write_registry(registry_path)

    summary = rollback_model(
        registry_path=registry_path,
        model_name="missing-model",
        target_version="foundation-config-v1",
        reason="high drift after deploy",
        report_path=tmp_path / "rollback.json",
    )

    assert summary == {
        "status": "failed",
        "model_name": "missing-model",
        "target_version": "foundation-config-v1",
        "error": "model not found",
    }


def test_production_rollback_model_cli_prints_json_summary(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    report_path = tmp_path / "reports" / "rollback.json"
    _write_registry(registry_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.production_patterns.rollback",
            "--registry-path",
            str(registry_path),
            "--model-name",
            "movielens-popularity",
            "--target-version",
            "foundation-config-v1",
            "--reason",
            "high drift after deploy",
            "--report-path",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "rolled_back"
    assert summary["target_version"] == "foundation-config-v1"
    assert report_path.exists()
