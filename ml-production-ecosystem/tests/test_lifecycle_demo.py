from pathlib import Path
import json

from production_patterns.lifecycle_demo import run_lifecycle_demo
from recommendation.train import get_active_model

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "recommendation"


def _write_config(tmp_path: Path, registry_path: Path) -> Path:
    version = "lifecycle-demo-v1"
    metrics_path = tmp_path / "artifacts" / "recommendation" / version / "metrics.json"
    input_schema = tmp_path / "schemas" / "input.json"
    output_schema = tmp_path / "schemas" / "output.json"
    input_schema.parent.mkdir(parents=True)
    input_schema.write_text(json.dumps({"title": "Input"}))
    output_schema.write_text(json.dumps({"title": "Output"}))
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

model_contract:
  input_schema_uri: {input_schema}
  output_schema_uri: {output_schema}
  task_type: ranking
  prediction_key: recommendations

artifacts:
  artifact_dir: {tmp_path / "artifacts"}

experiments:
  tracking_dir: {tmp_path / "experiments" / "runs"}
  run_id: {version}

registry:
  path: {registry_path}
  stage: candidate
  set_active: false

quality_gate:
  enabled: true
  metrics_path: {metrics_path}
  minimums:
    candidate_count: 1
    ratings_rows: 1
""".strip()
    )
    return config_path


def test_lifecycle_demo_writes_local_report_and_graph(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"
    output_path = tmp_path / "reports" / "lifecycle-demo.json"
    graph_path = tmp_path / "reports" / "lifecycle-demo.mmd"
    graph_html_path = tmp_path / "reports" / "lifecycle-demo.html"
    model_contract_path = tmp_path / "reports" / "model-contract-manifest.json"
    platform_report_path = tmp_path / "reports" / "platform-plan-validation.json"
    platform_plan_path = tmp_path / "platform-plan.yaml"
    approval_path = tmp_path / "reports" / "approval-decision.json"
    deployment_demo_path = tmp_path / "reports" / "deployment-demo.json"
    drift_report_path = tmp_path / "reports" / "drift-report.json"
    continual_learning_path = tmp_path / "reports" / "continual-learning-decision.json"

    platform_plan_path.write_text(
        """
provider: local
environment: development
resources:
  - kind: local-path
    name: model-artifacts
    uri: 01-foundation/artifacts
secrets: []
""".strip()
    )

    summary = run_lifecycle_demo(
        config_path=_write_config(tmp_path, registry_path),
        output_path=output_path,
        graph_path=graph_path,
        graph_html_path=graph_html_path,
        model_contract_path=model_contract_path,
        platform_plan_path=platform_plan_path,
        platform_report_path=platform_report_path,
        dataset_manifest_path=tmp_path / "reports" / "dataset-manifest.json",
        validation_report_path=tmp_path / "reports" / "offline-validation.json",
        approval_path=approval_path,
        deployment_demo_path=deployment_demo_path,
        drift_report_path=drift_report_path,
        continual_learning_path=continual_learning_path,
    )

    assert summary["status"] == "completed"
    assert summary["platform"]["status"] == "passed"
    assert summary["model_contract"]["status"] == "ready"
    assert summary["dataset"]["status"] == "ready"
    assert summary["training"]["version"] == "lifecycle-demo-v1"
    assert summary["offline_validation"]["status"] == "passed"
    assert summary["offline_validation"]["quality_gate"] == {"passed": True, "failures": []}
    assert summary["offline_validation"]["metrics"]["ratings_rows"] == 8.0
    assert summary["approval"]["status"] == "pending"
    assert summary["activation"] == {"set_active": False}
    assert summary["deployment_demo"]["status"] == "skipped"
    assert summary["continual_learning"]["action"] == "monitor"
    assert json.loads(output_path.read_text()) == summary
    assert json.loads(platform_report_path.read_text()) == summary["platform"]
    assert json.loads(approval_path.read_text()) == summary["approval"]
    assert json.loads(deployment_demo_path.read_text()) == summary["deployment_demo"]
    assert json.loads(drift_report_path.read_text()) == summary["drift"]
    assert json.loads(continual_learning_path.read_text()) == summary["continual_learning"]
    assert "flowchart LR" in graph_path.read_text()
    assert "mermaid.initialize" in graph_html_path.read_text()
    assert summary["graph"]["html_path"] == str(graph_html_path)


def test_lifecycle_demo_can_approve_and_activate_candidate(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry" / "models.json"

    summary = run_lifecycle_demo(
        config_path=_write_config(tmp_path, registry_path),
        approve=True,
        set_active=True,
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

    active = get_active_model(registry_path, "movielens-popularity")
    assert summary["approval"]["status"] == "approved"
    assert summary["activation"] == {"set_active": True}
    assert active is not None
    assert active["version"] == "lifecycle-demo-v1"
