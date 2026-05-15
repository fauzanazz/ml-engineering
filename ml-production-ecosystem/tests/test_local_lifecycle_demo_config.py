from pathlib import Path

import yaml

from production_patterns.lifecycle_demo import run_lifecycle_demo

ROOT = Path(__file__).resolve().parents[1]


def test_local_lifecycle_demo_config_runs_without_download(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs" / "local-lifecycle-demo.yaml").read_text())
    config["artifacts"]["artifact_dir"] = str(tmp_path / "artifacts")
    config["experiments"]["tracking_dir"] = str(tmp_path / "experiments" / "runs")
    config["registry"]["path"] = str(tmp_path / "registry" / "models.json")
    config["quality_gate"]["metrics_path"] = str(tmp_path / "artifacts" / "recommendation" / "local-demo-v1" / "metrics.json")
    config_path = tmp_path / "local-lifecycle-demo.yaml"
    config_path.write_text(yaml.safe_dump(config))

    summary = run_lifecycle_demo(
        config_path=config_path,
        output_path=tmp_path / "reports" / "lifecycle-demo.json",
        graph_path=tmp_path / "reports" / "lifecycle-demo.mmd",
        graph_html_path=tmp_path / "reports" / "lifecycle-demo.html",
        model_contract_path=tmp_path / "reports" / "model-contract-manifest.json",
        platform_report_path=tmp_path / "reports" / "platform-plan-validation.json",
        dataset_manifest_path=tmp_path / "reports" / "dataset-manifest.json",
        validation_report_path=tmp_path / "reports" / "offline-validation.json",
        approval_path=tmp_path / "reports" / "approval-decision.json",
        deployment_demo_path=tmp_path / "reports" / "deployment-demo.json",
        drift_report_path=tmp_path / "reports" / "drift-report.json",
        continual_learning_path=tmp_path / "reports" / "continual-learning-decision.json",
    )

    assert summary["status"] == "completed"
    assert summary["model_contract"]["status"] == "ready"
    assert summary["dataset"]["status"] == "ready"
    assert summary["offline_validation"]["status"] == "passed"
    assert summary["training"]["version"] == "local-demo-v1"
