from pathlib import Path
import json

from ml_production_ecosystem.production_patterns.offline_validation import build_offline_validation_report


def _write_config(tmp_path: Path, minimum: int) -> Path:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"candidate_count": 3, "ratings_rows": 8}))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
pipeline:
  name: demo-model
  version: v1
model:
  type: classifier
quality_gate:
  enabled: true
  metrics_path: {metrics_path}
  minimums:
    candidate_count: {minimum}
""".strip()
    )
    return config_path


def test_build_offline_validation_report_passes(tmp_path: Path) -> None:
    output_path = tmp_path / "offline-validation.json"

    report = build_offline_validation_report(_write_config(tmp_path, minimum=1), output_path)

    assert report["status"] == "passed"
    assert report["model"] == {"name": "demo-model", "version": "v1", "type": "classifier"}
    assert report["quality_gate"] == {"passed": True, "failures": []}
    assert report["metrics"] == {"candidate_count": 3.0, "ratings_rows": 8.0}
    assert json.loads(output_path.read_text()) == report


def test_build_offline_validation_report_fails(tmp_path: Path) -> None:
    report = build_offline_validation_report(_write_config(tmp_path, minimum=5), tmp_path / "report.json")

    assert report["status"] == "failed"
    assert report["quality_gate"] == {
        "passed": False,
        "failures": ["candidate_count 3.0 below minimum 5.0"],
    }
