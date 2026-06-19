"""Cross-provider platform apply command for local/cloud adapters."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ml_production_ecosystem.shared.platform import DeploymentExecution

DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/production-patterns")
PROVIDERS = ("local", "aws", "gcp", "azure")
ADAPTER_PATHS = {
    "local": Path("configs/platform/adapters/local/adapter.py"),
    "aws": Path("configs/platform/adapters/aws/adapter.py"),
    "gcp": Path("configs/platform/adapters/gcp/adapter.py"),
    "azure": Path("configs/platform/adapters/azure/adapter.py"),
}


def _load_adapter_module(project_root: Path, provider: str) -> Any:
    adapter_path = project_root / ADAPTER_PATHS[provider]
    spec = importlib.util.spec_from_file_location(
        f"platform_adapter_{provider}",
        adapter_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load adapter module: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _build_local_adapter(project_root: Path):
    module = _load_adapter_module(project_root, "local")
    config = module.LocalAdapterConfig(project_root=project_root)
    return module.LocalProviderAdapter(config)


def _build_cloud_adapter(project_root: Path, provider: str):
    module = _load_adapter_module(project_root, provider)
    adapter_type = getattr(module, f"{provider.capitalize()}ProviderAdapter")
    return adapter_type(
        plan_path=project_root
        / Path("configs/platform/iac")
        / provider
        / "platform-plan.yaml"
    )


def _provider_adapter(project_root: Path, provider: str):
    if provider == "local":
        return _build_local_adapter(project_root)
    return _build_cloud_adapter(project_root, provider)


def _read_plan(project_root: Path, provider: str) -> dict[str, Any]:
    plan_path = (
        project_root / Path("configs/platform/iac") / provider / "platform-plan.yaml"
    )
    data = yaml.safe_load(plan_path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def _format_template(template: str, **context: Any) -> str:
    try:
        return template.format(**context)
    except (KeyError, ValueError):
        return template


def _extract_apply_commands(
    plan: dict[str, Any],
    *,
    provider: str,
    environment: str,
    project_root: str,
) -> list[tuple[str, ...]]:
    apply_config = plan.get("apply")
    if not isinstance(apply_config, dict):
        return []

    commands_raw = apply_config.get("commands")
    if not isinstance(commands_raw, list):
        return []

    context = {
        "environment": environment,
        "provider": provider,
        "project_root": project_root,
    }
    commands: list[tuple[str, ...]] = []
    for raw_command in commands_raw:
        if isinstance(raw_command, str):
            try:
                expanded = _format_template(raw_command, **context)
                command = tuple(shlex.split(expanded))
            except ValueError:
                continue
        elif isinstance(raw_command, list) and all(
            isinstance(item, str) for item in raw_command
        ):
            expanded_items = [_format_template(item, **context) for item in raw_command]
            command = tuple(expanded_items)
        else:
            continue

        if command:
            commands.append(command)

    return commands


def _format_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _truncate_text(value: str, limit: int = 2048) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _run_apply_commands(
    project_root: Path,
    provider: str,
    environment: str,
) -> tuple[bool, list[dict[str, Any]]]:

    plan = _read_plan(project_root, provider)
    commands = _extract_apply_commands(
        plan,
        provider=provider,
        environment=environment,
        project_root=str(project_root),
    )

    if not commands:
        return False, [
            {
                "command": "(none)",
                "status": "skipped",
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "reason": "no apply commands configured in platform plan",
            }
        ]

    command_runs: list[dict[str, Any]] = []
    all_ok = True

    for command in commands:
        start_time = time.perf_counter()
        completed = subprocess.run(
            list(command),
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        stdout = _truncate_text(completed.stdout)
        stderr = _truncate_text(completed.stderr)
        run_status = "succeeded" if completed.returncode == 0 else "failed"
        all_ok = all_ok and (run_status == "succeeded")
        command_runs.append(
            {
                "command": _format_command(command),
                "status": run_status,
                "exit_code": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )

    return all_ok, command_runs


def _coerce_action_status(
    summary: dict[str, Any],
    *,
    status: str,
) -> None:
    for action in summary.get("actions", ()):
        action["status"] = status


def _with_apply_runtime(
    provider: str,
    summary: dict[str, Any],
    *,
    project_root: Path,
    environment: str,
) -> None:
    apply_ok, command_runs = _run_apply_commands(
        project_root,
        provider,
        environment,
    )

    summary["command_runs"] = command_runs
    summary["command_count"] = len(command_runs)
    summary["command_succeeded_count"] = sum(
        1 for run in command_runs if run["status"] == "succeeded"
    )
    summary["command_failed_count"] = sum(
        1 for run in command_runs if run["status"] == "failed"
    )

    if not apply_ok:
        summary["status"] = "failed"
        _coerce_action_status(summary, status="failed")
        summary["failure_reason"] = "runtime apply command failed"


def apply_platform(
    provider: str,
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    environment: str = "development",
    apply_changes: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if provider not in PROVIDERS:
        raise ValueError(f"unsupported provider: {provider}")

    adapter = _provider_adapter(project_root, provider)

    if provider == "local":
        summary = adapter.ensure_resources(environment)
        summary.setdefault("dry_run", False)
    else:
        execution = adapter.deploy(environment, dry_run=not apply_changes)
        summary = _serialize_execution(execution)
        if apply_changes:
            _with_apply_runtime(
                provider,
                summary,
                project_root=project_root,
                environment=environment,
            )

    summary["provider"] = provider
    summary["environment"] = environment

    output_path = output_path or (
        DEFAULT_OUTPUT_DIR / f"{provider}-platform-apply.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _serialize_execution(execution: DeploymentExecution) -> dict[str, Any]:
    return {
        "status": execution.status,
        "dry_run": execution.dry_run,
        "actions": [asdict(action) for action in execution.actions],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply platform resources through local/cloud adapters.")
    parser.add_argument("--provider", choices=("local", "aws", "gcp", "azure"), default="local")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--environment", default="development")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run deployment mode (no dry-run for provider adapters)",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    payload = apply_platform(
        args.provider,
        project_root=args.project_root,
        environment=args.environment,
        apply_changes=args.apply,
        output_path=args.output_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
