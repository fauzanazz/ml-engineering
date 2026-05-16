"""Validate local secret injection references without reading secret values."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path(".")
DEFAULT_OUTPUT_PATH = Path("02-production-patterns/reports/local-secret-injections.json")
LOCAL_PLAN_PATH = Path("04-platform-and-cloud/iac/local/platform-plan.yaml")
LOCAL_INJECTION_PATH = Path("04-platform-and-cloud/secrets/local/secret-injections.yaml")
REQUIRED_INJECTION_FIELDS = {
    "injection_target",
    "name",
    "policy_ref",
    "source_kind",
    "value_handling",
}


def validate_local_secret_injections(
    root: Path = DEFAULT_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    plan = _read_yaml(root / LOCAL_PLAN_PATH)
    manifest = _read_yaml(root / LOCAL_INJECTION_PATH)
    plan_secrets = _items(plan.get("secrets", []))
    injections = _items(manifest.get("injections", []))
    injections_by_target = {str(item.get("injection_target", "")): item for item in injections}
    failures = []

    if manifest.get("provider") != "local":
        failures.append("manifest provider must be local")
    if manifest.get("environment") != plan.get("environment"):
        failures.append("manifest environment must match local platform plan")

    for index, injection in enumerate(injections):
        missing_fields = REQUIRED_INJECTION_FIELDS - set(injection)
        if missing_fields:
            failures.append(f"injections[{index}] missing fields: {', '.join(sorted(missing_fields))}")
        if injection.get("value_handling") != "external-only":
            failures.append(f"injections[{index}] value_handling must be external-only")

    for secret in plan_secrets:
        injection_target = str(secret.get("injection_target", ""))
        injection = injections_by_target.get(injection_target)
        if injection is None:
            failures.append(f"missing local injection target: {injection_target}")
            continue
        if injection.get("name") != secret.get("name"):
            failures.append(f"secret name mismatch for {injection_target}")
        if injection.get("policy_ref") != secret.get("policy_ref"):
            failures.append(f"policy_ref mismatch for {injection_target}")

    report: dict[str, Any] = {
        "status": "passed" if not failures else "failed",
        "plan_path": str(LOCAL_PLAN_PATH),
        "injection_path": str(LOCAL_INJECTION_PATH),
        "required_targets": [str(secret.get("injection_target", "")) for secret in plan_secrets],
        "declared_targets": sorted(injections_by_target),
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def _items(value: object) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local secret injection references.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = validate_local_secret_injections(args.root, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
