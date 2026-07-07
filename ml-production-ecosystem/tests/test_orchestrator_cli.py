from argparse import Namespace
from pathlib import Path
import json

from ml_production_ecosystem.production_patterns import orchestrator_cli

from ml_production_ecosystem.production_patterns.orchestrator_cli import (
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


class _FakeAsk:
    """Mimics questionary's Question object: .ask() returns a fixed value."""

    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer


def _forbidden(name):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"questionary.{name} must not be called in this mode")

    return _raise


def test_select_uses_questionary_select_when_tty(monkeypatch) -> None:
    captured = {}

    def fake_select(label, choices, default):
        captured["label"] = label
        captured["values"] = [choice.value for choice in choices]
        captured["default"] = default
        return _FakeAsk("gcp")

    monkeypatch.setattr(orchestrator_cli, "_has_tui", lambda: True)
    monkeypatch.setattr(orchestrator_cli.questionary, "select", fake_select)

    result = orchestrator_cli._select("Provider", ("local", "aws", "gcp", "azure"), "local")

    assert result == "gcp"
    assert captured == {
        "label": "Provider",
        "values": ["local", "aws", "gcp", "azure"],
        "default": "local",
    }


def test_select_falls_back_to_default_when_questionary_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_cli, "_has_tui", lambda: True)
    monkeypatch.setattr(
        orchestrator_cli.questionary, "select", lambda label, choices, default: _FakeAsk(None)
    )

    result = orchestrator_cli._select("Provider", ("local", "aws"), "local")

    assert result == "local"


def test_select_uses_text_fallback_and_skips_questionary_when_not_tty(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_cli, "_has_tui", lambda: False)
    monkeypatch.setattr(orchestrator_cli.questionary, "select", _forbidden("select"))
    monkeypatch.setattr("builtins.input", lambda _prompt: "gcp")

    result = orchestrator_cli._select("Provider", ("local", "aws", "gcp"), "local")

    assert result == "gcp"


def test_prompt_infra_uses_questionary_checkbox_with_defaults_checked(monkeypatch) -> None:
    captured = {}

    def fake_checkbox(label, choices):
        captured["label"] = label
        captured["checked"] = {choice.value: choice.checked for choice in choices}
        return _FakeAsk(["registry", "monitoring"])

    monkeypatch.setattr(orchestrator_cli, "_has_tui", lambda: True)
    monkeypatch.setattr(orchestrator_cli.questionary, "checkbox", fake_checkbox)

    result = orchestrator_cli._prompt_infra(("registry",))

    assert result == ("registry", "monitoring")
    assert captured["label"] == "Components"
    assert captured["checked"]["registry"] is True
    assert captured["checked"]["monitoring"] is False


def test_prompt_infra_keeps_defaults_when_checkbox_selection_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_cli, "_has_tui", lambda: True)
    monkeypatch.setattr(
        orchestrator_cli.questionary, "checkbox", lambda label, choices: _FakeAsk(None)
    )

    result = orchestrator_cli._prompt_infra(("registry", "monitoring"))

    assert result == ("registry", "monitoring")


def test_prompt_infra_falls_back_to_yes_no_prompts_when_not_tty(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_cli, "_has_tui", lambda: False)
    monkeypatch.setattr(orchestrator_cli.questionary, "checkbox", _forbidden("checkbox"))
    monkeypatch.setattr("builtins.input", lambda _prompt: "y" if "registry" in _prompt else "n")

    result = orchestrator_cli._prompt_infra(())

    assert result == ("registry",)


def test_prompt_required_uses_questionary_text_and_reprompts_until_non_empty(monkeypatch, capsys) -> None:
    answers = iter(["", "   ", "Churn API"])
    monkeypatch.setattr(orchestrator_cli, "_has_tui", lambda: True)
    monkeypatch.setattr(orchestrator_cli.questionary, "text", lambda label: _FakeAsk(next(answers)))

    result = orchestrator_cli._prompt_required("Project name")

    assert result == "Churn API"
    assert capsys.readouterr().out.count("Project name is required.") == 2


def test_prompt_required_falls_back_to_input_when_not_tty(monkeypatch) -> None:
    monkeypatch.setattr(orchestrator_cli, "_has_tui", lambda: False)
    monkeypatch.setattr(orchestrator_cli.questionary, "text", _forbidden("text"))
    monkeypatch.setattr("builtins.input", lambda _prompt: "Churn API")

    result = orchestrator_cli._prompt_required("Project name")

    assert result == "Churn API"
