from pathlib import Path
import json
import sys

from ml_production_ecosystem.production_patterns.retraining import run_retraining
from ml_production_ecosystem.recommendation.train import get_active_model


def _write_cnn_segmentation_config(tmp_path: Path) -> Path:
    model_name = "cnn-segmentation"
    version = "seg-v1"
    metrics_path = tmp_path / "artifacts" / "cnn_segmentation" / version / "metrics.json"
    artifact_path = tmp_path / "artifacts" / "cnn_segmentation" / version
    summary_path = tmp_path / "artifacts" / "reports" / "cnn-segmentation-training-summary.json"
    train_script = tmp_path / "train_cnn_segmentation.py"
    train_script.write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        f"artifact_path = Path({str(artifact_path)!r})\n"
        f"metrics_path = Path({str(metrics_path)!r})\n"
        f"summary_path = Path({str(summary_path)!r})\n"
        "artifact_path.mkdir(parents=True, exist_ok=True)\n"
        "metrics_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "summary_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "metrics_path.write_text(json.dumps({'mean_iou': 0.91, 'dice_coefficient': 0.88}))\n"
        "summary_path.write_text(json.dumps({\n"
        f"    'model_name': {model_name!r},\n"
        f"    'version': {version!r},\n"
        "    'artifact_uri': str(artifact_path),\n"
        "    'metrics_uri': str(metrics_path),\n"
        "}))\n"
    )

    config = f"""training:
  type: pytorch
  framework: pytorch
  command:
    - {sys.executable}
    - {train_script}
  summary_path: {summary_path}
quality_gate:
  enabled: true
  metrics_path: {metrics_path}
  minimums:
    mean_iou: 0.85
    dice_coefficient: 0.8
registry:
  path: {tmp_path / 'registry' / 'models.json'}
"""
    config_path = tmp_path / "cnn-segmentation-command.yaml"
    config_path.write_text(config)
    return config_path




def _write_cnn_segmentation_loss_config(tmp_path: Path) -> Path:
    model_name = "cnn-segmentation"
    version = "seg-loss-v1"
    metrics_path = tmp_path / "artifacts" / "cnn_segmentation" / version / "metrics.json"
    artifact_path = tmp_path / "artifacts" / "cnn_segmentation" / version
    summary_path = tmp_path / "artifacts" / "reports" / "cnn-segmentation-loss-summary.json"
    train_script = tmp_path / "train_cnn_segmentation_loss.py"
    train_script.write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        f"artifact_path = Path({str(artifact_path)!r})\n"
        f"metrics_path = Path({str(metrics_path)!r})\n"
        f"summary_path = Path({str(summary_path)!r})\n"
        "artifact_path.mkdir(parents=True, exist_ok=True)\n"
        "metrics_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "summary_path.parent.mkdir(parents=True, exist_ok=True)\n\n"
        "loss_history = []\n"
        "for epoch in range(5):\n"
        "    # Minimal synthetic training step for segmentation-like binary mask objective\n"
        "    loss = 0.55 / (epoch + 1)\n"
        "    loss_history.append(loss)\n"
        "\n"
        "# Keep deterministic and monotonic decreasing.\n"
        "metrics = {\n"
        "    'mean_iou': 0.89,\n"
        "    'dice_coefficient': 0.86,\n"
        "    'loss': loss_history[-1],\n"
        "    'loss_history': loss_history,\n"
        "}\n"
        "metrics_path.write_text(json.dumps(metrics))\n"
        "summary_path.write_text(json.dumps({\n"
        f"    'model_name': {model_name!r},\n"
        f"    'version': {version!r},\n"
        "    'artifact_uri': str(artifact_path),\n"
        "    'metrics_uri': str(metrics_path),\n"
        "}))\n"
    )

    config = f"""training:
  type: pytorch
  framework: pytorch
  command:
    - {sys.executable}
    - {train_script}
  summary_path: {summary_path}
quality_gate:
  enabled: true
  metrics_path: {metrics_path}
  minimums:
    loss: 0.1
registry:
  path: {tmp_path / 'registry' / 'models.json'}
"""
    config_path = tmp_path / "cnn-segmentation-loss.yaml"
    config_path.write_text(config)
    return config_path


def test_run_retraining_supports_cnn_segmentation_pytorch_command_config(tmp_path: Path) -> None:
    config_path = _write_cnn_segmentation_config(tmp_path)
    registry_path = tmp_path / "registry" / "models.json"

    summary = run_retraining(
        config_path,
        set_active=True,
        registry_path=registry_path,
        model_name="cnn-segmentation",
        require_quality_gate=True,
    )

    assert summary["status"] == "completed"
    assert summary["model_name"] == "cnn-segmentation"
    assert summary["version"] == "seg-v1"
    assert summary["set_active"] is True
    assert summary["quality_gate"] == {"passed": True, "failures": []}

    active = get_active_model(registry_path, "cnn-segmentation")
    assert active is not None
    assert active["version"] == "seg-v1"
    assert summary["artifact_uri"].endswith("artifacts/cnn_segmentation/seg-v1")
    assert summary["metrics_uri"].endswith("artifacts/cnn_segmentation/seg-v1/metrics.json")
    assert active["artifact_uri"].endswith("artifacts/cnn_segmentation/seg-v1")
    assert json.loads((Path(summary["metrics_uri"]).read_text()))["mean_iou"] == 0.91


def test_run_retraining_supports_cnn_segmentation_loss_curve(tmp_path: Path) -> None:
    config_path = _write_cnn_segmentation_loss_config(tmp_path)
    registry_path = tmp_path / "registry" / "models.json"

    summary = run_retraining(
        config_path,
        set_active=True,
        registry_path=registry_path,
        model_name="cnn-segmentation",
        require_quality_gate=True,
    )

    assert summary["status"] == "completed"
    assert summary["model_name"] == "cnn-segmentation"
    assert summary["version"] == "seg-loss-v1"
    assert summary["set_active"] is True
    assert summary["quality_gate"] == {"passed": True, "failures": []}

    metrics = json.loads(Path(summary["metrics_uri"]).read_text())
    assert "loss_history" in metrics
    assert metrics["loss_history"][0] > metrics["loss_history"][-1]
    assert metrics["loss"] == metrics["loss_history"][-1]
