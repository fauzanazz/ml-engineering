#!/usr/bin/env python3
"""Runtime command dispatcher for cloud provider platform apply hooks.

The command reads provider-specific command templates from environment variables and
executes them during `production-apply-platform --apply`.

Environment:
- PLATFORM_APPLY_AWS_COMMAND
- PLATFORM_APPLY_GCP_COMMAND
- PLATFORM_APPLY_AZURE_COMMAND

Each variable is rendered with Python str.format using:
- provider
- environment
- project_root

If no provider command is configured, the hook exits with status 0 and prints a
placeholder message.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower()


def _command_env_var(provider: str) -> str:
    return f"PLATFORM_APPLY_{provider.upper()}_COMMAND"


def _build_default_message(provider: str, environment: str) -> str:
    return f"Applying {provider} platform plan to environment: {environment}"


def _resolve_command(provider: str, environment: str, project_root: str) -> tuple[str, list[str]]:
    env_name = _command_env_var(provider)
    template = os.environ.get(env_name)
    if template is None:
        template = f"python -c \"print('{_build_default_message(provider, environment)}')\""

    try:
        command_text = template.format(
            provider=provider,
            environment=environment,
            project_root=project_root,
        )
    except Exception:
        raise RuntimeError(
            f"failed to format command template for {provider!r} from {env_name!r}"
        ) from None

    try:
        command = shlex.split(command_text)
    except ValueError as exc:
        raise RuntimeError(f"failed to split command for {provider!r}: {exc}") from exc

    return env_name, command


def run(provider: str, *, environment: str, project_root: str) -> int:
    provider = _normalize_provider(provider)

    if not provider:
        print("provider is required", file=sys.stderr)
        return 2

    env_name, command = _resolve_command(provider, environment, project_root)

    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            check=False,
        )
    except FileNotFoundError:
        print(
            f"[platform-apply-runtime] command missing for {provider} (environment variable {env_name} may reference unavailable executable)",
            file=sys.stderr,
        )
        return 127
    except Exception as exc:
        print(
            f"[platform-apply-runtime] failed to execute command for {provider}: {exc}",
            file=sys.stderr,
        )
        return 1

    if result.returncode == 0 and os.environ.get(env_name) is None:
        print(
            f"[platform-apply-runtime] no {env_name} configured for {provider};"
            f" executed fallback message command"
        )

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cloud provider apply hook command.")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--environment", default="development")
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()

    return run(args.provider, environment=args.environment, project_root=args.project_root)


if __name__ == "__main__":
    sys.exit(main())
