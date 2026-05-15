from argparse import Namespace
from pathlib import Path
import json

from production_patterns.orchestrator_cli import (
    build_parser,
    print_human_status,
    run_doctor,
    run_explain,
    run_status,
)


def test_parser_defaults_to_interactive_menu() -> None:
    args = build_parser().parse_args([])

    assert args.handler.__name__ == "run_menu"


def test_parser_registers_quickstart_command() -> None:
    args = build_parser().parse_args(["quickstart", "--json"])

    assert args.handler.__name__ == "run_quickstart"
    assert args.config == Path("configs/local-lifecycle-demo.yaml")
    assert args.json is True


def test_status_prints_human_next_step(tmp_path: Path, capsys) -> None:
    report_dir = tmp_path / "reports"
    output_path = tmp_path / "status.json"
    report_dir.mkdir()

    result = run_status(Namespace(report_dir=report_dir, output_path=output_path, json=False))

    assert result == 0
    output = capsys.readouterr().out
    assert "ML Production Status" in output
    assert "Next: uv run mle quickstart" in output


def test_status_prints_json(tmp_path: Path, capsys) -> None:
    report_dir = tmp_path / "reports"
    output_path = tmp_path / "status.json"
    report_dir.mkdir()

    result = run_status(Namespace(report_dir=report_dir, output_path=output_path, json=True))

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "incomplete"


def test_explain_summarizes_missing_state(tmp_path: Path, capsys) -> None:
    report_dir = tmp_path / "reports"
    output_path = tmp_path / "status.json"
    report_dir.mkdir()

    result = run_explain(Namespace(report_dir=report_dir, output_path=output_path))

    assert result == 0
    output = capsys.readouterr().out
    assert "Lifecycle incomplete" in output
    assert "Missing:" in output


def test_doctor_fails_outside_project(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    result = run_doctor(Namespace())

    assert result == 1
    output = capsys.readouterr().out
    assert "[missing] pyproject.toml" in output
