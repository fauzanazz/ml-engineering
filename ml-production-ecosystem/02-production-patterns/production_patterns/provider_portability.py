"""Validate platform plans can move across providers without core changes."""

from pathlib import Path
import argparse
import json

from shared.platform import PlatformPlanAdapter

DEFAULT_ROOT = Path(".")
DEFAULT_OUTPUT_PATH = Path("02-production-patterns/reports/provider-portability.json")
PROVIDERS = ("local", "aws", "gcp", "azure")
REQUIRED_RESOURCE_NAMES = {"model-artifacts", "model-registry", "prediction-logs"}
REQUIRED_SECRET_TARGETS = {"MODEL_REGISTRY_TOKEN", "LOCAL_MODEL_REGISTRY_TOKEN"}

def _plan_path(root: Path, provider: str) -> Path:
    return root / "04-platform-and-cloud" / "iac" / provider / "platform-plan.yaml"

def _resource_names(root: Path, provider: str) -> set[str]:
    plan = PlatformPlanAdapter(_plan_path(root, provider)).plan("development")
    return {resource.name for resource in plan.resources}

def _secret_targets(root: Path, provider: str) -> set[str]:
    plan = PlatformPlanAdapter(_plan_path(root, provider)).plan("development")
    return {secret.injection_target for secret in plan.secrets}

def validate_provider_portability(
    root: Path = DEFAULT_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    resource_names_by_provider = {provider: sorted(_resource_names(root, provider)) for provider in PROVIDERS}
    secret_targets_by_provider = {provider: sorted(_secret_targets(root, provider)) for provider in PROVIDERS}
    failures = []

    for provider, resource_names in resource_names_by_provider.items():
        missing = sorted(REQUIRED_RESOURCE_NAMES - set(resource_names))
        if missing:
            failures.append(f"{provider} missing resources: {', '.join(missing)}")

    for provider, secret_targets in secret_targets_by_provider.items():
        if not (set(secret_targets) & REQUIRED_SECRET_TARGETS):
            failures.append(f"{provider} missing model registry secret injection target")

    cloud_resource_sets = {
        provider: set(resource_names_by_provider[provider])
        for provider in ("aws", "gcp", "azure")
    }
    if len({tuple(sorted(names)) for names in cloud_resource_sets.values()}) != 1:
        failures.append("cloud provider resource names differ")

    report = {
        "status": "passed" if not failures else "failed",
        "providers": list(PROVIDERS),
        "required_resource_names": sorted(REQUIRED_RESOURCE_NAMES),
        "resource_names_by_provider": resource_names_by_provider,
        "secret_targets_by_provider": secret_targets_by_provider,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate provider plans preserve portable resource contracts.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = validate_provider_portability(args.root, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
