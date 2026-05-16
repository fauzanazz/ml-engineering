"""E2E rollback proof on command-trained generic classifier (no recommender)."""

from pathlib import Path
import json
import shutil

import yaml

from ml_production_ecosystem.production_patterns.retraining import run_retraining
from ml_production_ecosystem.production_patterns.rollback import rollback_model
from ml_production_ecosystem.shared.model_storage.registry import get_active_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "generic-classifier-command.yaml"


def _portable_config(tmp_path: Path, version: str) -> Path:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    data_path = ROOT / "examples" / "samples" / "generic_classifier" / "data.csv"
    train_path = ROOT / "examples" / "samples" / "generic_classifier" / "train.py"
    artifact_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "reports" / f"summary-{version}.json"

    config["pipeline"]["version"] = version
    config["training"]["command"] = [
        shutil.which("python") or "python",
        str(train_path),
        "--data-path",
        str(data_path),
        "--artifact-dir",
        str(artifact_dir),
        "--summary-path",
        str(summary_path),
        "--version",
        version,
    ]
    config["training"]["summary_path"] = str(summary_path)
    config["registry"]["path"] = str(tmp_path / "registry" / "models.json")
    config["quality_gate"]["metrics_path"] = str(
        artifact_dir / "tiny-threshold-classifier" / version / "metrics.json"
    )

    config_path = tmp_path / f"config-{version}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def test_rollback_generic_classifier_to_previous_version(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    report_path = tmp_path / "reports" / "rollback.json"

    good_version = "generic-classifier-good-v1"
    bad_version = "generic-classifier-bad-v2"

    run_retraining(_portable_config(tmp_path, good_version), set_active=True, require_quality_gate=True)
    run_retraining(_portable_config(tmp_path, bad_version), set_active=True, require_quality_gate=True)

    active_before = get_active_model(registry_path, "tiny-threshold-classifier")
    assert active_before is not None
    assert active_before["version"] == bad_version

    summary = rollback_model(
        registry_path=registry_path,
        model_name="tiny-threshold-classifier",
        target_version=good_version,
        reason="bad-v2 caused regression",
        report_path=report_path,
    )

    assert summary["status"] == "rolled_back"
    assert summary["model_name"] == "tiny-threshold-classifier"
    assert summary["target_version"] == good_version

    active_after = get_active_model(registry_path, "tiny-threshold-classifier")
    assert active_after is not None
    assert active_after["version"] == good_version

    report = json.loads(report_path.read_text())
    assert report["status"] == "rolled_back"
    assert report["reason"] == "bad-v2 caused regression"


def test_rollback_generic_classifier_rejects_unknown_version(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    report_path = tmp_path / "reports" / "rollback-fail.json"

    version = "generic-classifier-only-v1"
    run_retraining(_portable_config(tmp_path, version), set_active=True, require_quality_gate=True)

    summary = rollback_model(
        registry_path=registry_path,
        model_name="tiny-threshold-classifier",
        target_version="generic-classifier-ghost",
        reason="unknown version",
        report_path=report_path,
    )

    assert summary["status"] == "failed"
    assert summary["error"] == "target version not found"

    active = get_active_model(registry_path, "tiny-threshold-classifier")
    assert active is not None
    assert active["version"] == version
