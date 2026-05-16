from pathlib import Path
import json
import shutil

import yaml

from ml_production_ecosystem.production_patterns.model_contract_manifest import build_model_contract_manifest
from ml_production_ecosystem.production_patterns.retraining import run_retraining
from ml_production_ecosystem.recommendation.train import get_active_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "generic-classifier-command.yaml"

def _portable_config(tmp_path: Path) -> Path:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    data_path = ROOT / "examples" / "samples" / "generic_classifier" / "data.csv"
    train_path = ROOT / "examples" / "samples" / "generic_classifier" / "train.py"
    artifact_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "reports" / "generic-classifier-training-summary.json"
    version = "generic-classifier-test-v1"

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

    config_path = tmp_path / "generic-classifier-command.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path

def test_generic_classifier_command_config_declares_model_contract() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["model_contract"] == {
        "input_schema_uri": "schemas/generic_classifier/input.json",
        "output_schema_uri": "schemas/generic_classifier/output.json",
        "task_type": "classification",
        "prediction_key": "label",
    }
    assert config["training"]["type"] == "command"

def test_generic_classifier_model_contract_manifest_is_ready(tmp_path: Path) -> None:
    manifest = build_model_contract_manifest(CONFIG_PATH, tmp_path / "manifest.json")

    assert manifest["status"] == "ready"
    assert manifest["contract"]["task_type"] == "classification"

def test_generic_classifier_command_training_can_activate_without_recommender_code(tmp_path: Path) -> None:
    config_path = _portable_config(tmp_path)
    registry_path = tmp_path / "registry" / "models.json"

    summary = run_retraining(config_path, set_active=True, require_quality_gate=True)
    active = get_active_model(registry_path, "tiny-threshold-classifier")
    metrics = json.loads(Path(summary["metrics_uri"]).read_text())

    assert summary["status"] == "completed"
    assert summary["model_name"] == "tiny-threshold-classifier"
    assert summary["set_active"] is True
    assert summary["quality_gate"] == {"passed": True, "failures": []}
    assert active is not None
    assert active["version"] == "generic-classifier-test-v1"
    assert metrics["accuracy"] >= 0.8
