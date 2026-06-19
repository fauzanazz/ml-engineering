"""Validate provider-neutral platform/IaC plan references."""

from pathlib import Path
import argparse
import json
from typing import Any

import yaml

DEFAULT_PLAN_PATH = Path("configs/platform/iac/local/platform-plan.yaml")
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/platform-plan-validation.json")
SECRET_VALUE_KEYS = {"value", "secret", "password", "token_value", "api_key"}


def _read_plan(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def _missing_resource_fields(resources: list[Any]) -> list[str]:
    missing = []
    for index, resource in enumerate(resources):
        if not isinstance(resource, dict):
            missing.append(f"resources[{index}] must be object")
            continue
        for field in ("kind", "name", "uri"):
            if not resource.get(field):
                missing.append(f"resources[{index}].{field}")
    return missing


def _secret_violations(secrets: list[Any]) -> list[str]:
    violations = []
    for index, secret in enumerate(secrets):
        if not isinstance(secret, dict):
            violations.append(f"secrets[{index}] must be object")
            continue
        for field in ("provider", "name", "injection_target", "policy_ref"):
            if not secret.get(field):
                violations.append(f"secrets[{index}].{field}")
        forbidden = SECRET_VALUE_KEYS & set(secret)
        for key in sorted(forbidden):
            violations.append(f"secrets[{index}] contains forbidden value key {key}")
    return violations


def validate_platform_plan(
    plan_path: Path = DEFAULT_PLAN_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    plan = _read_plan(plan_path)
    resources = plan.get("resources", [])
    secrets = plan.get("secrets", [])
    resource_failures = _missing_resource_fields(resources if isinstance(resources, list) else [])
    secret_failures = _secret_violations(secrets if isinstance(secrets, list) else [])
    failures = []
    if not plan.get("provider"):
        failures.append("provider")
    if not plan.get("environment"):
        failures.append("environment")
    failures.extend(resource_failures)
    failures.extend(secret_failures)

    report = {
        "status": "passed" if not failures else "failed",
        "plan_path": str(plan_path),
        "provider": plan.get("provider"),
        "environment": plan.get("environment"),
        "resource_count": len(resources) if isinstance(resources, list) else 0,
        "secret_reference_count": len(secrets) if isinstance(secrets, list) else 0,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate provider-neutral platform plan.")
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = validate_platform_plan(args.plan_path, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
