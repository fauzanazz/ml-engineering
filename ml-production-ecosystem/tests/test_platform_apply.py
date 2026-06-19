from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import yaml

from ml_production_ecosystem.production_patterns.platform_apply import apply_platform

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_HOOK_PATH = ROOT / "scripts" / "platform_apply_runtime.py"


def _copy_local_context(tmp_path: Path) -> None:
    shutil.copytree(
        ROOT / "configs" / "platform" / "adapters" / "local",
        tmp_path / "configs" / "platform" / "adapters" / "local",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        ROOT / "configs" / "platform" / "iac" / "local",
        tmp_path / "configs" / "platform" / "iac" / "local",
        dirs_exist_ok=True,
    )
    _copy_platform_apply_runtime(tmp_path)


def _copy_platform_apply_runtime(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNTIME_HOOK_PATH, scripts_dir / "platform_apply_runtime.py")


def _copy_provider_context(tmp_path: Path, provider: str) -> None:
    shutil.copytree(
        ROOT / "configs" / "platform" / "adapters" / provider,
        tmp_path / "configs" / "platform" / "adapters" / provider,
        dirs_exist_ok=True,
    )
    shutil.copytree(
        ROOT / "configs" / "platform" / "iac" / provider,
        tmp_path / "configs" / "platform" / "iac" / provider,
        dirs_exist_ok=True,
    )
    _copy_platform_apply_runtime(tmp_path)


def test_apply_platform_local_provider_returns_ready_and_writes_report(tmp_path: Path) -> None:
    _copy_local_context(tmp_path)
    output_path = tmp_path / "artifacts" / "platform-apply.json"

    summary = apply_platform(
        "local",
        project_root=tmp_path,
        environment="development",
        output_path=output_path,
    )

    assert summary["provider"] == "local"
    assert summary["status"] == "ready"
    assert summary["dry_run"] is False
    assert summary == json.loads(output_path.read_text())
    assert (tmp_path / "artifacts" / "foundation").is_dir()
    assert (tmp_path / "registry").is_dir()
    assert "LOCAL_MODEL_REGISTRY_TOKEN" in summary["secret_references"]


def test_apply_platform_cloud_provider_dry_run_preview(tmp_path: Path) -> None:
    for provider in ("aws", "gcp", "azure"):
        _copy_provider_context(tmp_path, provider)

    output_path = tmp_path / "artifacts" / "platform-apply-aws.json"
    summary = apply_platform(
        "aws",
        project_root=tmp_path,
        environment="development",
        apply_changes=False,
        output_path=output_path,
    )

    assert summary["provider"] == "aws"
    assert summary["status"] == "planned"
    assert summary["dry_run"] is True
    assert summary["environment"] == "development"
    assert len(summary["actions"]) == 5
    assert output_path.read_text()


def test_apply_platform_cloud_provider_apply_mode(tmp_path: Path) -> None:
    _copy_provider_context(tmp_path, "gcp")

    summary = apply_platform(
        "gcp",
        project_root=tmp_path,
        environment="development",
        apply_changes=True,
    )

    assert summary["provider"] == "gcp"
    assert summary["status"] == "applied"
    assert summary["dry_run"] is False
    assert summary["environment"] == "development"
    assert summary["command_count"] == 1
    assert summary["command_failed_count"] == 0
    assert summary["command_runs"][0]["status"] == "succeeded"
    assert "Applying gcp platform plan to environment: development" in summary["command_runs"][0]["stdout"]


def test_apply_platform_cloud_apply_uses_command_template_environment(tmp_path: Path) -> None:
    _copy_provider_context(tmp_path, "aws")
    env = os.environ.copy()
    env["PLATFORM_APPLY_AWS_COMMAND"] = "python -c \"print('runtime:{provider}:{environment}:{project_root}')\""

    os.environ.update(env)
    try:
        summary = apply_platform(
            "aws",
            project_root=tmp_path,
            environment="development",
            apply_changes=True,
        )
    finally:
        os.environ.pop("PLATFORM_APPLY_AWS_COMMAND", None)

    assert summary["provider"] == "aws"
    assert summary["command_runs"][0]["status"] == "succeeded"
    assert "runtime:aws:development:" in summary["command_runs"][0]["stdout"]


def test_apply_platform_cloud_apply_templates_environment(tmp_path: Path) -> None:
    _copy_provider_context(tmp_path, "aws")

    summary = apply_platform(
        "aws",
        project_root=tmp_path,
        environment="development",
        apply_changes=True,
    )

    assert summary["provider"] == "aws"
    assert "Applying aws platform plan to environment: development" in summary["command_runs"][0]["stdout"]


def test_apply_platform_cli_writes_plan(tmp_path: Path) -> None:
    _copy_provider_context(tmp_path, "azure")
    output_path = tmp_path / "artifacts" / "platform-apply.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_production_ecosystem.production_patterns.platform_apply",
            "--provider",
            "azure",
            "--project-root",
            str(tmp_path),
            "--output-path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["provider"] == "azure"
    assert payload["dry_run"] is True
    assert result.stdout


def test_apply_platform_apply_mode_fails_when_command_hook_fails(tmp_path: Path) -> None:
    _copy_provider_context(tmp_path, "aws")
    plan_path = tmp_path / "configs" / "platform" / "iac" / "aws" / "platform-plan.yaml"
    plan_data = yaml.safe_load(plan_path.read_text())
    plan_data.setdefault("apply", {})["commands"] = [["python", "-c", "import sys; sys.exit(1)"]]
    plan_path.write_text(yaml.safe_dump(plan_data))

    summary = apply_platform(
        "aws",
        project_root=tmp_path,
        environment="development",
        apply_changes=True,
    )

    assert summary["provider"] == "aws"
    assert summary["status"] == "failed"
    assert summary["command_failed_count"] == 1
    assert all(action["status"] == "failed" for action in summary["actions"])