from argparse import Namespace
from pathlib import Path
import importlib
import importlib.util
import json
import subprocess
import sys
import tomllib

from fastapi.testclient import TestClient
import pytest
import yaml

from ml_production_ecosystem.production_patterns import orchestrator_cli
from ml_production_ecosystem.production_patterns.orchestrator_cli import build_parser, run_new
from ml_production_ecosystem.production_patterns.scaffold import (
    PRESET_DEFAULTS,
    SUPPORTED_INFRA,
    SUPPORTED_PRESETS,
    ScaffoldRequest,
    TEMPLATE_ROOT,
    package_name_from_project,
    scaffold_project,
)


def test_package_name_from_project_normalizes_python_package() -> None:
    assert package_name_from_project("House Prices 2026!") == "house_prices_2026"
    assert package_name_from_project("123 churn") == "ml_123_churn"


def test_scaffold_project_rejects_empty_project_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Project name"):
        scaffold_project(
            ScaffoldRequest(
                preset="kaggle",
                name="---",
                target=tmp_path / "empty",
            )
        )


def test_scaffold_project_rejects_non_empty_target(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    (target / "README.md").write_text("keep")

    with pytest.raises(FileExistsError, match="not empty"):
        scaffold_project(
            ScaffoldRequest(
                preset="kaggle",
                name="House Prices",
                target=target,
            )
        )


@pytest.mark.parametrize("preset", SUPPORTED_PRESETS)
def test_scaffold_project_writes_common_boilerplate(tmp_path: Path, preset: str) -> None:
    target = tmp_path / preset

    result = scaffold_project(
        ScaffoldRequest(
            preset=preset,
            name="House Prices",
            target=target,
        )
    )

    assert result.package_name == "house_prices"
    assert (target / "README.md").exists()
    assert (target / "pyproject.toml").exists()
    assert (target / "configs" / "project.yaml").read_text().splitlines()[:2] == [
        "project: House Prices",
        f"preset: {preset}",
    ]
    assert (target / "data" / "README.md").exists()
    assert (target / "docs" / "runbook.md").exists()
    assert (target / "house_prices" / "__init__.py").exists()
    assert (target / "tests" / "test_scaffold.py").exists()
    assert all("{{" not in path.read_text() for path in result.written_paths)




def test_scaffold_project_writes_modular_axes(tmp_path: Path) -> None:
    target = tmp_path / "modular"

    result = scaffold_project(
        ScaffoldRequest(
            preset="existing-model-wrapper",
            name="Existing ASR",
            target=target,
            task="speech-to-text",
            model_type="whisper",
            backend="external-command",
            infra=("registry", "quality-gate", "monitoring"),
        )
    )

    config = (target / "ml-struct.yaml").read_text()
    checklist = (target / "docs" / "infra-checklist.md").read_text()
    assert result.task == "speech-to-text"
    assert "model_type: whisper" in config
    assert "backend: external-command" in config
    assert "- [ ] registry" in checklist


def test_scaffold_project_applies_provider_flag_and_writes_ml_struct_yaml(tmp_path: Path) -> None:
    target = tmp_path / "provider-aws"

    result = scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Provider AWS",
            target=target,
            provider="aws",
        )
    )

    struct_lines = (target / "ml-struct.yaml").read_text().splitlines()
    assert result.provider == "aws"
    assert "provider: aws" in struct_lines


def test_scaffold_project_writes_pypi_safe_package_name_in_pyproject(tmp_path: Path) -> None:
    target = tmp_path / "pyproject-name"

    result = scaffold_project(
        ScaffoldRequest(
            preset="kaggle",
            name="House Prices Advanced",
            target=target,
        )
    )

    data = tomllib.loads((target / "pyproject.toml").read_text())
    assert data["project"]["name"] == result.package_name
    assert " " not in data["project"]["name"]


@pytest.mark.parametrize(
    ("task", "task_type", "prediction_key"),
    [
        ("classification", "classification", "label"),
        ("regression", "regression", "value"),
        ("object-detection", "object_detection", "detections"),
        ("segmentation", "segmentation", "mask"),
        ("text-generation", "text_generation", "text"),
    ],
)
def test_existing_model_wrapper_bootstrap_supports_common_task_output_forms(tmp_path: Path, task: str, task_type: str, prediction_key: str) -> None:
    target = tmp_path / f"wrapper-{task.replace('-', '_')}"
    result = scaffold_project(
        ScaffoldRequest(
            preset="existing-model-wrapper",
            name=f"Task {task}",
            target=target,
            task=task,
        )
    )

    config_text = (target / "configs" / "project.yaml").read_text()
    assert f"task_type: {task_type}" in config_text
    assert f"prediction_key: {prediction_key}" in config_text
    assert "training:" in config_text
    assert f"{result.package_name}.train" in config_text
    assert "reports/training-summary.json" in config_text
    assert (target / "schemas" / task / "input.json").exists()
    assert (target / "schemas" / task / "output.json").exists()


def test_existing_model_wrapper_bootstrap_includes_task_train_example(tmp_path: Path) -> None:
    target = tmp_path / "wrapper-train-example"
    result = scaffold_project(
        ScaffoldRequest(
            preset="existing-model-wrapper",
            name="My Existing Project",
            target=target,
            task="classification",
        )
    )

    train_path = target / result.package_name / "train.py"
    train_text = train_path.read_text()
    assert train_path.exists()
    assert 'TASK_TYPE = "classification"' in train_text
    assert 'PREDICTION_KEY = "label"' in train_text

    summary_path = target / "artifacts" / "reports" / "bootstrap-summary.json"
    run_result = subprocess.run(
        [
            sys.executable,
            str(train_path),
            "--summary-path",
            str(summary_path),
        ],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0
    summary = json.loads(summary_path.read_text())
    assert summary["model_name"] == result.package_name
    metrics_data = json.loads((summary_path.parent / "metrics.json").read_text())
    assert "loss" in metrics_data
    assert "loss_history" in metrics_data

    state_path = target / "model_state.json"
    state_path.write_text(json.dumps({"weights": [0.05, -0.02, 0.01, 0.03], "bias": 0.04}, indent=2) + "\n")
    loaded_summary = target / "artifacts" / "reports" / "loaded-summary.json"
    run_result = subprocess.run(
        [
            sys.executable,
            str(train_path),
            "--summary-path",
            str(loaded_summary),
            "--model-state",
            str(state_path),
        ],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0
    loaded_metrics = json.loads((loaded_summary.parent / "metrics.json").read_text())
    assert "loss" in loaded_metrics


def test_text_generation_task_train_example_emits_generation_metrics(tmp_path: Path) -> None:
    target = tmp_path / "wrapper-text-generation"
    result = scaffold_project(
        ScaffoldRequest(
            preset="existing-model-wrapper",
            name="Text Generation Project",
            target=target,
            task="text-generation",
        )
    )

    train_path = target / result.package_name / "train.py"
    summary_path = target / "artifacts" / "reports" / "text-gen-summary.json"
    subprocess.run(
        [
            sys.executable,
            str(train_path),
            "--summary-path",
            str(summary_path),
        ],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads((summary_path.parent / "metrics.json").read_text())

    assert "perplexity" in metrics
    assert "bleu" in metrics
    assert "latency_ms" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "loss" in metrics


def test_train_script_normalizes_directory_summary_path_to_file(tmp_path: Path) -> None:
    target = tmp_path / "wrapper-summary-dir"
    result = scaffold_project(
        ScaffoldRequest(
            preset="existing-model-wrapper",
            name="Summary Dir Project",
            target=target,
            task="classification",
        )
    )

    train_path = target / result.package_name / "train.py"
    summary_dir = target / "artifacts" / "manual-summary-dir"
    summary_dir.mkdir(parents=True)
    summary_file = summary_dir / "training-summary.json"
    subprocess.run(
        [
            sys.executable,
            str(train_path),
            "--summary-path",
            str(summary_dir),
        ],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(summary_file.read_text())
    assert summary["model_name"] == result.package_name
    assert (summary_dir / "metrics.json").exists()


def test_served_model_scaffold_includes_api_and_dockerfile(tmp_path: Path) -> None:
    target = tmp_path / "served"

    scaffold_project(
        ScaffoldRequest(
            preset="served-model",
            name="Churn API",
            target=target,
        )
    )

    assert (target / "churn_api" / "api.py").exists()
    assert "uvicorn" in (target / "Dockerfile").read_text()


def test_served_model_metadata_matches_default_generated_paths(tmp_path: Path) -> None:
    target = tmp_path / "served-contract"
    result = scaffold_project(
        ScaffoldRequest(preset="served-model", name="Churn API", target=target)
    )
    metadata = yaml.safe_load(
        (TEMPLATE_ROOT / "served-model" / "template.yaml").read_text()
        .replace("{{package_name}}", result.package_name)
    )

    assert set(metadata["contract"]["generated_paths"]) == {
        path.relative_to(target).as_posix() for path in result.written_paths
    }


def test_served_model_scaffold_generates_a_working_health_and_predict_api(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "served-live"

    scaffold_project(
        ScaffoldRequest(
            preset="served-model",
            name="Churn API",
            target=target,
        )
    )
    monkeypatch.syspath_prepend(str(target))

    api = importlib.import_module("churn_api.api")
    client = TestClient(api.app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    health_body = health_response.json()
    assert health_body["status"] == "ok"

    predict_response = client.post("/predict", json={"features": {"a": 1.0, "b": 2.0}})
    assert predict_response.status_code == 200
    predict_body = predict_response.json()
    assert isinstance(predict_body["prediction"], bool)
    assert predict_body["model_name"] == health_body["model_name"]
    assert predict_body["model_version"] == health_body["model_version"]

    rejected = client.post("/predict", json={})
    assert rejected.status_code == 422


def test_served_model_scaffold_dockerfile_cmd_starts_uvicorn_not_pytest(tmp_path: Path) -> None:
    target = tmp_path / "served-docker"

    scaffold_project(
        ScaffoldRequest(
            preset="served-model",
            name="Churn API",
            target=target,
        )
    )

    dockerfile_lines = (target / "Dockerfile").read_text().splitlines()
    cmd_lines = [line for line in dockerfile_lines if line.strip().startswith("CMD")]

    assert len(cmd_lines) == 1
    assert "uvicorn" in cmd_lines[0]
    assert "churn_api.api:app" in cmd_lines[0]
    assert "pytest" not in cmd_lines[0]

def test_asr_served_model_scaffold_includes_contract_and_api(tmp_path: Path) -> None:
    target = tmp_path / "banking-asr"

    scaffold_project(
        ScaffoldRequest(
            preset="asr-served-model",
            name="Banking ASR",
            target=target,
        )
    )

    config = (target / "configs" / "project.yaml").read_text()
    assert "task_type: speech_to_text" in config
    assert "wer: 0.25" in config
    assert (target / "schemas" / "asr" / "input.json").exists()
    assert (target / "banking_asr" / "api.py").exists()
    assert (target / "banking_asr" / "train.py").exists()




@pytest.mark.parametrize(
    ("preset", "package_file", "expected_text"),
    [
        ("generic-classifier", "predict.py", "positive"),
        ("recommendation", "rank.py", "recommend"),
        ("batch-inference", "batch.py", "predict_batch"),
        ("existing-model-wrapper", "adapter.py", "write_summary"),
        ("llm-post-training", "evaluate.py", "pass_rate"),
    ],
)
def test_flexible_gap_presets_include_main_seam(
    tmp_path: Path,
    preset: str,
    package_file: str,
    expected_text: str,
) -> None:
    target = tmp_path / preset

    scaffold_project(
        ScaffoldRequest(
            preset=preset,
            name="Flexible Project",
            target=target,
        )
    )

    assert expected_text in (target / "flexible_project" / package_file).read_text()
    assert (target / "configs" / "project.yaml").exists()


def test_llm_post_training_dataset_build_examples_has_no_todo_placeholder(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "llm-post-training"

    result = scaffold_project(
        ScaffoldRequest(
            preset="llm-post-training",
            name="Reasoning Tutor",
            target=target,
        )
    )

    dataset_path = target / result.package_name / "dataset.py"
    dataset_text = dataset_path.read_text()
    assert "TODO" not in dataset_text

    monkeypatch.syspath_prepend(str(target))
    dataset_module = importlib.import_module(f"{result.package_name}.dataset")
    examples = dataset_module.build_examples(["seed prompt"])
    assert examples == [{"prompt": "seed prompt", "answer": "synthetic response for: seed prompt"}]


def test_enterprise_pipeline_scaffold_includes_quality_gate(tmp_path: Path) -> None:
    target = tmp_path / "enterprise"

    scaffold_project(
        ScaffoldRequest(
            preset="enterprise-pipeline",
            name="Fraud Pipeline",
            target=target,
        )
    )

    assert "approval" in (target / "fraud_pipeline" / "pipeline.py").read_text()
    assert (target / "fraud_pipeline" / "quality_gate.py").exists()


def test_scaffold_project_resolves_retraining_dependencies(tmp_path: Path) -> None:
    target = tmp_path / "retraining-deps"

    result = scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Retraining Deps",
            target=target,
            infra=("retraining",),
        )
    )

    assert set(result.dependency_additions) == {
        "training",
        "evaluation",
        "deployment",
        "registry",
        "quality-gate",
        "monitoring",
    }
    for component in result.dependency_additions + ("retraining",):
        assert component in result.infra


def test_scaffold_project_writes_component_files_for_selected_infra(tmp_path: Path) -> None:
    target = tmp_path / "component-files"

    result = scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Component Files",
            target=target,
            infra=("ci", "retraining"),
        )
    )

    package_dir = target / result.package_name
    assert (target / ".github" / "workflows" / "ci.yml").exists()
    assert (package_dir / "deployment.py").exists()
    assert (target / "tests" / "test_deployment.py").exists()
    assert (package_dir / "evaluate.py").exists()
    assert (target / "tests" / "test_evaluate.py").exists()
    assert (target / "orchestration" / "monitoring_job.py").exists()
    assert (target / "orchestration" / "retraining_job.py").exists()


def test_scaffold_project_accepts_metaflow_backend(tmp_path: Path) -> None:
    target = tmp_path / "metaflow-pipeline"

    result = scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Metaflow Pipeline",
            target=target,
            backend="metaflow",
            infra=("retraining",),
        )
    )

    assert result.backend == "metaflow"
    assert "retraining" in result.infra
    assert "training" in result.infra


def test_scaffold_project_writes_metaflow_backend_into_ml_struct_yaml(tmp_path: Path) -> None:
    target = tmp_path / "metaflow-struct"

    scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Metaflow Pipeline",
            target=target,
            backend="metaflow",
            infra=("retraining",),
        )
    )

    struct_lines = (target / "ml-struct.yaml").read_text().splitlines()
    assert "backend: metaflow" in struct_lines


def test_scaffold_project_writes_metaflow_orchestration_adapter_skeleton(tmp_path: Path) -> None:
    target = tmp_path / "metaflow-adapter"

    result = scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Metaflow Pipeline",
            target=target,
            backend="metaflow",
            infra=("retraining",),
        )
    )

    adapter_path = target / "orchestration" / "metaflow_retraining_flow.py"
    assert adapter_path.exists()
    assert adapter_path in result.written_paths

    source = adapter_path.read_text()
    assert "FlowSpec" in source
    assert "orchestration/retraining_job.py" in source
    assert "orchestration/monitoring_job.py" in source
    assert "production-scheduled-retrain" not in source
    assert "production-monitor" not in source

    # Import-safety: the skeleton must load even though metaflow is not
    # installed in this environment, mirroring the Airflow DAG skeleton.
    spec = importlib.util.spec_from_file_location("metaflow_retraining_flow", adapter_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_scaffold_project_metaflow_backend_skips_airflow_adapter(tmp_path: Path) -> None:
    target = tmp_path / "metaflow-only"

    result = scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Metaflow Only",
            target=target,
            backend="metaflow",
            infra=("retraining",),
        )
    )

    assert result.backend == "metaflow"
    assert (target / "orchestration" / "metaflow_retraining_flow.py").exists()
    assert not (target / "orchestration" / "airflow_retraining_dag.py").exists()


def test_scaffold_metaflow_orchestration_jobs_run_standalone_and_write_json_reports(tmp_path: Path) -> None:
    target = tmp_path / "metaflow-jobs"

    scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Metaflow Jobs",
            target=target,
            backend="metaflow",
            infra=("retraining",),
        )
    )

    retraining_job = target / "orchestration" / "retraining_job.py"
    monitoring_job = target / "orchestration" / "monitoring_job.py"
    assert retraining_job.exists()
    assert monitoring_job.exists()

    config_path = target / "configs" / "project.yaml"
    retraining_report_path = target / "artifacts" / "reports" / "scheduled-retraining.json"
    retraining_result = subprocess.run(
        [
            sys.executable,
            str(retraining_job),
            "--config",
            str(config_path),
            "--output-path",
            str(retraining_report_path),
            "--set-active",
            "--require-quality-gate",
        ],
        cwd=target,
        capture_output=True,
        text=True,
    )
    assert retraining_result.returncode == 0, retraining_result.stderr
    assert "metaflow" not in retraining_result.stderr.lower()
    assert "airflow" not in retraining_result.stderr.lower()
    retraining_report = json.loads(retraining_report_path.read_text())
    assert retraining_report["status"] == "completed"
    assert retraining_report["set_active"] is True

    monitoring_report_path = target / "artifacts" / "reports" / "monitoring.json"
    monitoring_result = subprocess.run(
        [
            sys.executable,
            str(monitoring_job),
            "--input-path",
            str(retraining_report_path),
            "--output-path",
            str(monitoring_report_path),
        ],
        cwd=target,
        capture_output=True,
        text=True,
    )
    assert monitoring_result.returncode == 0, monitoring_result.stderr
    monitoring_report = json.loads(monitoring_report_path.read_text())
    assert monitoring_report["status"] == "healthy"
    assert all(check["passed"] for check in monitoring_report["checks"])


def test_scaffold_metaflow_retraining_job_promotes_active_model_after_local_training(tmp_path: Path) -> None:
    target = tmp_path / "metaflow-retraining"
    package_name = package_name_from_project("Metaflow Retraining")

    scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Metaflow Retraining",
            target=target,
            backend="metaflow",
            infra=("retraining",),
        )
    )

    retraining_job = target / "orchestration" / "retraining_job.py"
    config_path = target / "configs" / "project.yaml"
    report_path = target / "artifacts" / "reports" / "scheduled-retraining.json"
    result = subprocess.run(
        [
            sys.executable,
            str(retraining_job),
            "--config",
            str(config_path),
            "--output-path",
            str(report_path),
            "--set-active",
            "--require-quality-gate",
        ],
        cwd=target,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed"
    assert report["model_name"] == package_name
    assert report["artifact_uri"] == "models/local-dev"
    assert report["metrics_uri"] == "reports/metrics.json"
    assert report["quality_gate"]["status"] == "passed"
    assert report["quality_gate"]["failures"] == []
    assert report["quality_gate"]["metrics"]["accuracy"] == 1.0
    assert report["active_model_uri"] == "artifacts/active-model.json"

    active_model_path = target / "artifacts" / "active-model.json"
    assert active_model_path.exists()
    active_model = json.loads(active_model_path.read_text())
    assert active_model["model_name"] == package_name
    assert active_model["artifact_uri"] == "models/local-dev"


def test_scaffold_metaflow_retraining_job_fails_and_skips_promotion_on_quality_gate_breach(tmp_path: Path) -> None:
    target = tmp_path / "metaflow-retraining-fail"

    scaffold_project(
        ScaffoldRequest(
            preset="generic-classifier",
            name="Metaflow Retraining Fail",
            target=target,
            backend="metaflow",
            infra=("retraining",),
        )
    )

    config_path = target / "configs" / "project.yaml"
    config_text = config_path.read_text()
    assert "accuracy: 0.8" in config_text
    config_path.write_text(config_text.replace("accuracy: 0.8", "accuracy: 1.5"))

    retraining_job = target / "orchestration" / "retraining_job.py"
    report_path = target / "artifacts" / "reports" / "scheduled-retraining.json"
    result = subprocess.run(
        [
            sys.executable,
            str(retraining_job),
            "--config",
            str(config_path),
            "--output-path",
            str(report_path),
            "--set-active",
            "--require-quality-gate",
        ],
        cwd=target,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed_quality_gate"
    assert report["quality_gate"]["status"] == "failed"
    assert report["quality_gate"]["failures"] == ["accuracy=1.0 below minimum 1.5"]
    assert "active_model_uri" not in report
    assert not (target / "artifacts" / "active-model.json").exists()


def test_enterprise_pipeline_default_airflow_backend_skips_metaflow_adapter(tmp_path: Path) -> None:
    target = tmp_path / "enterprise-airflow"

    result = scaffold_project(
        ScaffoldRequest(
            preset="enterprise-pipeline",
            name="Fraud Pipeline",
            target=target,
        )
    )

    assert result.backend == "airflow"
    assert "retraining" in result.infra
    airflow_adapter = target / "orchestration" / "airflow_retraining_dag.py"
    assert airflow_adapter.exists()
    assert "FlowSpec" not in airflow_adapter.read_text()
    assert not (target / "orchestration" / "metaflow_retraining_flow.py").exists()


def test_parser_registers_new_command(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "new",
            "--preset",
            "kaggle",
            "--name",
            "House Prices",
            "--target",
            str(tmp_path / "house-prices"),
        ]
    )

    assert args.handler.__name__ == "run_new"
    assert args.preset == "kaggle"


def test_parser_allows_interactive_new_command() -> None:
    args = build_parser().parse_args(["new"])

    assert args.handler.__name__ == "run_new"
    assert args.preset is None
    assert args.name is None
    assert args.project_name is None
    assert args.target is None


def test_parser_accepts_positional_project_name() -> None:
    args = build_parser().parse_args(["new", "banking-asr", "--preset", "asr-served-model"])

    assert args.project_name == "banking-asr"
    assert args.preset == "asr-served-model"


def test_parser_accepts_metaflow_backend() -> None:
    args = build_parser().parse_args(
        [
            "new",
            "metaflow-project",
            "--preset",
            "generic-classifier",
            "--backend",
            "metaflow",
        ]
    )

    assert args.backend == "metaflow"


def test_parser_accepts_provider_flag() -> None:
    args = build_parser().parse_args(
        [
            "new",
            "provider-project",
            "--preset",
            "generic-classifier",
            "--provider",
            "gcp",
        ]
    )

    assert args.provider == "gcp"


def test_parser_accepts_no_input_and_list_presets() -> None:
    no_input_args = build_parser().parse_args(["new", "banking-asr", "--preset", "asr-served-model", "--no-input"])
    list_args = build_parser().parse_args(["new", "--list-presets"])

    assert no_input_args.no_input is True
    assert list_args.list_presets is True


def test_create_main_accepts_project_name_without_new(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create-ml-struct",
            "churn-api",
            "--preset",
            "served-model",
            "--no-input",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        orchestrator_cli.create_main()

    assert exit_info.value.code == 0
    assert "Project Scaffolded" in capsys.readouterr().out
    assert (tmp_path / "churn-api" / "churn_api" / "api.py").exists()


def test_run_new_prints_next_command(tmp_path: Path, capsys) -> None:
    target = tmp_path / "house-prices"

    result = run_new(
        Namespace(
            preset="kaggle",
            name="House Prices",
            target=target,
            no_input=True,
            force=False,
        )
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Project Scaffolded" in output
    assert "Package: house_prices" in output
    assert f"Next: cd {target.resolve()} && uv run pytest" in output


def test_run_new_prompts_for_missing_values(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    answers = iter(["served-model", "Churn API", "", "", "", "", *([""] * len(SUPPORTED_INFRA))])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    result = run_new(
        Namespace(
            preset=None,
            name=None,
            target=None,
            force=False,
        )
    )

    output = capsys.readouterr().out
    assert result == 0
    for label in ("Task options:", "Model type options:", "Backend options:", "Provider options:", "Components:"):
        assert label in output
    _, _, default_backend, _ = PRESET_DEFAULTS["served-model"]
    assert "Preset: served-model" in output
    assert f"Backend: {default_backend}" in output
    assert (tmp_path / "churn-api" / "churn_api" / "api.py").exists()


class _FakeAsk:
    """Mimics questionary's Question object: .ask() returns a fixed value."""

    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer


def _forbidden(name):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"questionary.{name} must not be called")

    return _raise


def test_run_new_uses_questionary_prompts_when_tty(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator_cli, "_has_tui", lambda: True)
    monkeypatch.setattr(orchestrator_cli.questionary, "text", lambda label: _FakeAsk("Churn API"))

    def fake_select(label, choices, default):
        chosen = {"Project type": "served-model", "Task": "regression", "Provider": "gcp"}
        return _FakeAsk(chosen.get(label, default))

    monkeypatch.setattr(orchestrator_cli.questionary, "select", fake_select)
    monkeypatch.setattr(
        orchestrator_cli.questionary, "checkbox", lambda label, choices: _FakeAsk(["registry"])
    )

    result = run_new(Namespace(preset=None, name=None, target=None, force=False))

    output = capsys.readouterr().out
    assert result == 0
    assert "Task: regression" in output
    assert "Provider: gcp" in output
    components_line = next(line for line in output.splitlines() if line.startswith("Components:"))
    assert "registry" in components_line
    assert "docker" not in components_line
    assert (tmp_path / "churn-api" / "churn_api" / "api.py").exists()


def test_run_new_no_input_never_invokes_any_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator_cli.questionary, "select", _forbidden("select"))
    monkeypatch.setattr(orchestrator_cli.questionary, "checkbox", _forbidden("checkbox"))
    monkeypatch.setattr(orchestrator_cli.questionary, "text", _forbidden("text"))
    monkeypatch.setattr("builtins.input", _forbidden("input"))

    result = run_new(
        Namespace(
            preset="served-model",
            name="Churn API",
            target=None,
            no_input=True,
            list_presets=False,
            force=False,
        )
    )

    assert result == 0
    assert (tmp_path / "churn-api" / "churn_api" / "api.py").exists()


def test_run_new_uses_safe_default_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    answers = iter(["asr-served-model", "Banking ASR", "", "", "", "", *([""] * len(SUPPORTED_INFRA))])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    result = run_new(
        Namespace(
            preset=None,
            name=None,
            target=None,
            force=False,
        )
    )

    assert result == 0
    assert (tmp_path / "banking-asr" / "banking_asr" / "api.py").exists()


def test_run_new_uses_positional_name_and_default_target(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    result = run_new(
        Namespace(
            project_name="Banking ASR",
            preset="asr-served-model",
            name=None,
            target=None,
            no_input=True,
            list_presets=False,
            force=False,
        )
    )

    output = capsys.readouterr().out
    assert result == 0
    default_task, _, default_backend, _ = PRESET_DEFAULTS["asr-served-model"]
    assert f"Task: {default_task}" in output
    assert f"Backend: {default_backend}" in output
    assert (tmp_path / "banking-asr" / "banking_asr" / "api.py").exists()


def test_run_new_lists_presets(capsys) -> None:
    result = run_new(
        Namespace(
            project_name=None,
            preset=None,
            name=None,
            target=None,
            no_input=False,
            list_presets=True,
            force=False,
        )
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "existing-model-wrapper: wrap existing train/evaluate commands" in output
    assert "llm-post-training: reasoning/LLM data" in output


def test_help_includes_examples(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["new", "--help"])

    output = capsys.readouterr().out
    assert "Examples:" in output
    assert "ml-struct new banking-asr --preset asr-served-model" in output


def test_generated_asr_package_imports_and_writes_summary(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "banking-asr"
    scaffold_project(
        ScaffoldRequest(
            preset="asr-served-model",
            name="Banking ASR",
            target=target,
        )
    )
    monkeypatch.syspath_prepend(str(target))

    train = importlib.import_module("banking_asr.train")
    summary = train.write_training_summary(target / "reports" / "training-summary.json")

    assert summary["model_name"] == "banking_asr"
    assert (target / "reports" / "training-summary.json").exists()
    assert (target / "reports" / "metrics.json").exists()
