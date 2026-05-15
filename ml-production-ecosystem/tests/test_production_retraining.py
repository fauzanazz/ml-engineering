from pathlib import Path
import json
import subprocess
import sys

from production_patterns.retraining import run_retraining
from recommendation.train import get_active_model

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "recommendation"


def _write_retraining_config(
    tmp_path: Path,
    registry_path: Path,
    version: str = "foundation-config-v1",
    quality_gate_block: str = "",
) -> Path:
    config_path = tmp_path / "foundation-recommender.yaml"
    config_path.write_text(
        f"""
pipeline:
  name: foundation-recommender
  version: {version}

dataset:
  ratings_path: {FIXTURE_DIR / "ratings.csv"}
  movies_path: {FIXTURE_DIR / "movies.csv"}

model:
  type: popularity
  hyperparams:
    min_rating: 4.0

artifacts:
  artifact_dir: {tmp_path / "artifacts"}

experiments:
  tracking_dir: {tmp_path / "experiments" / "runs"}
  run_id: {version}

registry:
  path: {registry_path}
  stage: candidate
  set_active: false
{quality_gate_block}
""".strip()
    )
    return config_path


def test_run_retraining_returns_summary_without_active_update(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    config_path = _write_retraining_config(tmp_path, registry_path)

    summary = run_retraining(config_path, set_active=False)

    assert summary == {
        "status": "completed",
        "model_name": "movielens-popularity",
        "version": "foundation-config-v1",
        "artifact_uri": str(tmp_path / "artifacts" / "recommendation" / "foundation-config-v1"),
        "metrics_uri": str(tmp_path / "artifacts" / "recommendation" / "foundation-config-v1" / "metrics.json"),
        "set_active": False,
        "quality_gate": {"passed": True, "failures": []},
    }
    assert get_active_model(registry_path, "movielens-popularity") is None

def test_run_retraining_supports_command_based_model_training(tmp_path: Path) -> None:
    summary_path = tmp_path / "command-summary.json"
    metrics_path = tmp_path / "metrics.json"
    artifact_path = tmp_path / "artifact"
    command = (
        "import json, pathlib; "
        f"pathlib.Path({str(metrics_path)!r}).write_text(json.dumps({{'accuracy': 0.9}})); "
        f"pathlib.Path({str(artifact_path)!r}).mkdir(); "
        f"pathlib.Path({str(summary_path)!r}).write_text(json.dumps("
        "{'model_name':'custom-classifier','version':'v1','artifact_uri':%r,'metrics_uri':%r}))"
        % (str(artifact_path), str(metrics_path))
    )
    config_path = tmp_path / "command-model.yaml"
    config_path.write_text(
        f"""
training:
  type: command
  command:
    - {sys.executable}
    - -c
    - "{command}"
  summary_path: {summary_path}
quality_gate:
  enabled: true
  metrics_path: {metrics_path}
  minimums:
    accuracy: 0.8
""".strip()
    )

    summary = run_retraining(config_path, require_quality_gate=True, model_name="custom-classifier")

    assert summary == {
        "status": "completed",
        "model_name": "custom-classifier",
        "version": "v1",
        "artifact_uri": str(artifact_path),
        "metrics_uri": str(metrics_path),
        "set_active": False,
        "quality_gate": {"passed": True, "failures": []},
    }


def test_run_retraining_can_set_active_model(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    config_path = _write_retraining_config(tmp_path, registry_path)

    summary = run_retraining(config_path, set_active=True, registry_path=registry_path, model_name="movielens-popularity")

    active = get_active_model(registry_path, "movielens-popularity")
    assert summary["set_active"] is True
    assert active is not None
    assert active["version"] == "foundation-config-v1"


def test_run_retraining_pass_quality_gate_can_set_active_model(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    metrics_path = tmp_path / "artifacts" / "recommendation" / "foundation-config-v1" / "metrics.json"
    config_path = _write_retraining_config(
        tmp_path,
        registry_path,
        quality_gate_block=f"""

quality_gate:
  enabled: true
  metrics_path: {metrics_path}
  minimums:
    candidate_count: 1
    ratings_rows: 1
""".rstrip(),
    )

    summary = run_retraining(
        config_path,
        set_active=True,
        registry_path=registry_path,
        model_name="movielens-popularity",
        require_quality_gate=True,
    )

    active = get_active_model(registry_path, "movielens-popularity")
    assert summary["status"] == "completed"
    assert summary["set_active"] is True
    assert summary["quality_gate"] == {"passed": True, "failures": []}
    assert active is not None
    assert active["version"] == "foundation-config-v1"


def test_run_retraining_failed_quality_gate_does_not_set_active_model(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    metrics_path = tmp_path / "artifacts" / "recommendation" / "foundation-config-v1" / "metrics.json"
    config_path = _write_retraining_config(
        tmp_path,
        registry_path,
        quality_gate_block=f"""

quality_gate:
  enabled: true
  metrics_path: {metrics_path}
  minimums:
    candidate_count: 999
    ratings_rows: 1
""".rstrip(),
    )

    summary = run_retraining(
        config_path,
        set_active=True,
        registry_path=registry_path,
        model_name="movielens-popularity",
        require_quality_gate=True,
    )

    assert summary["status"] == "failed_quality_gate"
    assert summary["set_active"] is False
    assert summary["quality_gate"] == {
        "passed": False,
        "failures": ["candidate_count 4.0 below minimum 999.0"],
    }
    assert get_active_model(registry_path, "movielens-popularity") is None


def test_run_retraining_missing_quality_gate_keeps_current_behavior(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    config_path = _write_retraining_config(tmp_path, registry_path)

    summary = run_retraining(
        config_path,
        set_active=True,
        registry_path=registry_path,
        model_name="movielens-popularity",
        require_quality_gate=True,
    )

    active = get_active_model(registry_path, "movielens-popularity")
    assert summary["status"] == "completed"
    assert summary["quality_gate"] == {"passed": True, "failures": []}
    assert active is not None
    assert active["version"] == "foundation-config-v1"


def test_production_retrain_cli_prints_summary(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    config_path = _write_retraining_config(tmp_path, registry_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "production_patterns.retraining",
            "--config",
            str(config_path),
            "--set-active",
            "--registry-path",
            str(registry_path),
            "--model-name",
            "movielens-popularity",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "completed"
    assert summary["model_name"] == "movielens-popularity"
    assert summary["version"] == "foundation-config-v1"
    assert summary["set_active"] is True
