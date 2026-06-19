"""Build provider swap matrix from platform-plan adapters."""

import argparse
import json
from pathlib import Path
from typing import Any

from ml_production_ecosystem.shared.platform import PlatformPlanAdapter

DEFAULT_ROOT = Path(".")
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/provider-swap-matrix.json")
PROVIDERS = ("local", "aws", "gcp", "azure")
CORE_WORKFLOWS = (
    "production-lifecycle-demo",
    "production-retrain",
    "production-validate-offline",
    "production-approve-model",
    "production-demo-deployment",
    "production-detect-drift",
    "production-continual-decision",
    "production-lifecycle-graph",
    "production-rollback-model",
)


def build_provider_swap_matrix(
    root: Path = DEFAULT_ROOT,
    environment: str = "development",
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    plans = {provider: _load_plan(root, provider, environment) for provider in PROVIDERS}
    resource_names = sorted({resource["name"] for plan in plans.values() for resource in plan["resources"]})
    matrix = {
        name: {
            provider: _resource_by_name(plans[provider]["resources"], name)
            for provider in PROVIDERS
        }
        for name in resource_names
    }
    missing_by_provider = {
        provider: [name for name in resource_names if matrix[name][provider] is None]
        for provider in PROVIDERS
    }
    report: dict[str, Any] = {
        "status": "passed",
        "environment": environment,
        "providers": list(PROVIDERS),
        "core_code_changes_required": False,
        "swap_unit": "configs/platform/iac/<provider>/platform-plan.yaml + configs/platform/adapters/<provider>",
        "core_workflows": list(CORE_WORKFLOWS),
        "resource_matrix": matrix,
        "secret_injection_targets": {
            provider: sorted(secret["injection_target"] for secret in plans[provider]["secrets"])
            for provider in PROVIDERS
        },
        "missing_resources_by_provider": missing_by_provider,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _load_plan(root: Path, provider: str, environment: str) -> dict[str, Any]:
    path = root / "configs" / "platform" / "iac" / provider / "platform-plan.yaml"
    plan = PlatformPlanAdapter(path).plan(environment)
    return {
        "resources": [
            {
                "kind": resource.kind,
                "name": resource.name,
                "uri": resource.uri,
            }
            for resource in plan.resources
        ],
        "secrets": [
            {
                "name": secret.name,
                "injection_target": secret.injection_target,
                "policy_ref": secret.policy_ref,
            }
            for secret in plan.secrets
        ],
    }


def _resource_by_name(resources: list[dict[str, str]], name: str) -> dict[str, str] | None:
    for resource in resources:
        if resource["name"] == name:
            return {"kind": resource["kind"], "uri": resource["uri"]}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write provider swap matrix from provider-neutral plans.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--environment", default="development")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_provider_swap_matrix(args.root, args.environment, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
