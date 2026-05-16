"""Validate IaC secret policy references resolve to policies-as-code."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path(".")
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/policy-reference-validation.json")
PLAN_ROOT = Path("configs/platform/iac")
POLICY_ROOT = Path("configs/platform/policies")
REQUIRED_POLICY_FIELDS = {
    "actions",
    "effect",
    "injection_targets",
    "name",
    "policy_ref",
    "provider",
    "resources",
    "scope",
    "value_handling",
}


def validate_policy_references(
    root: Path = DEFAULT_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    policies = _load_policies(root)
    references = _load_policy_references(root)
    failures = []

    for reference in references:
        policy = policies.get(reference["policy_ref"])
        if policy is None:
            failures.append(f"missing policy_ref: {reference['policy_ref']}")
            continue
        if "invalid" in policy:
            failures.append(f"invalid policy_ref {reference['policy_ref']}: {policy['invalid']}")
            continue
        if policy["provider"] != reference["provider"]:
            failures.append(f"provider mismatch for {reference['policy_ref']}")
        if reference["injection_target"] not in policy["injection_targets"]:
            failures.append(f"injection target mismatch for {reference['policy_ref']}")

    report: dict[str, Any] = {
        "status": "passed" if not failures else "failed",
        "policy_count": len(policies),
        "reference_count": len(references),
        "policy_refs": sorted(policies),
        "references": references,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _load_policies(root: Path) -> dict[str, dict[str, Any]]:
    policies = {}
    for path in sorted((root / POLICY_ROOT).glob("*/*.yaml")):
        payload = _read_yaml(path)
        missing = REQUIRED_POLICY_FIELDS - set(payload)
        if missing:
            policy_ref = str(payload.get("policy_ref", path.relative_to(root)))
            policies[policy_ref] = {
                "provider": str(payload.get("provider", "")),
                "injection_targets": [],
                "invalid": f"missing fields: {', '.join(sorted(missing))}",
            }
            continue
        policy_ref = str(payload["policy_ref"])
        policies[policy_ref] = {
            "provider": str(payload["provider"]),
            "injection_targets": [str(target) for target in _items(payload["injection_targets"])],
        }
    return policies


def _load_policy_references(root: Path) -> list[dict[str, str]]:
    references = []
    for path in sorted((root / PLAN_ROOT).glob("*/platform-plan.yaml")):
        plan = _read_yaml(path)
        provider = str(plan.get("provider", ""))
        for item in _items(plan.get("secrets", [])):
            references.append(
                {
                    "provider": provider,
                    "policy_ref": str(item.get("policy_ref", "")),
                    "injection_target": str(item.get("injection_target", "")),
                    "plan_path": str(path.relative_to(root)),
                }
            )
    return references


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def _items(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate policy references in platform plans.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = validate_policy_references(args.root, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
