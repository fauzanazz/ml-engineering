from argparse import Namespace
from pathlib import Path
import importlib

import pytest

from ml_production_ecosystem.production_patterns.orchestrator_cli import build_parser, run_new
from ml_production_ecosystem.production_patterns.scaffold import (
    SUPPORTED_PRESETS,
    ScaffoldRequest,
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


def test_parser_accepts_no_input_and_list_presets() -> None:
    no_input_args = build_parser().parse_args(["new", "banking-asr", "--preset", "asr-served-model", "--no-input"])
    list_args = build_parser().parse_args(["new", "--list-presets"])

    assert no_input_args.no_input is True
    assert list_args.list_presets is True


def test_run_new_prints_next_command(tmp_path: Path, capsys) -> None:
    target = tmp_path / "house-prices"

    result = run_new(
        Namespace(
            preset="kaggle",
            name="House Prices",
            target=target,
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
    answers = iter(["served-model", "Churn API", "churn-api", "", "", "", ""])
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
    assert "Preset: served-model" in output
    assert (tmp_path / "churn-api" / "churn_api" / "api.py").exists()

def test_run_new_uses_safe_default_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    answers = iter(["asr-served-model", "Banking ASR", "", "", "", "", ""])
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


def test_run_new_uses_positional_name_and_default_target(tmp_path: Path, monkeypatch) -> None:
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

    assert result == 0
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
