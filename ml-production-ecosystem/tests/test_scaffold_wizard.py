from argparse import Namespace
from pathlib import Path

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
    assert args.target is None


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
    answers = iter(["served-model", "Churn API", "churn-api"])
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
