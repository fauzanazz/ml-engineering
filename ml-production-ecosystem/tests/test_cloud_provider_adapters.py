"""Cloud provider adapter tests."""

from pathlib import Path
import importlib.util

from ml_production_ecosystem.shared.platform import ProviderAdapter

ROOT = Path(__file__).resolve().parents[1]
PLATFORM_ROOT = ROOT / "configs" / "platform"

PROVIDER_CLASSES = {
    "aws": "AwsProviderAdapter",
    "gcp": "GcpProviderAdapter",
    "azure": "AzureProviderAdapter",
}


def test_cloud_provider_adapters_load_platform_plans() -> None:
    for provider, class_name in PROVIDER_CLASSES.items():
        module = _load_adapter(provider)
        adapter: ProviderAdapter = getattr(module, class_name)(
            PLATFORM_ROOT / "iac" / provider / "platform-plan.yaml"
        )

        plan = adapter.plan("development")

        assert plan.provider == provider
        assert {resource.name for resource in plan.resources} == {
            "model-artifacts",
            "model-images",
            "model-serving",
            "prediction-logs",
            "model-registry",
        }
        assert plan.secrets[0].provider == provider
        assert plan.secrets[0].injection_target == "MODEL_REGISTRY_TOKEN"


def test_cloud_provider_adapters_preview_deployment_execution() -> None:
    for provider, class_name in PROVIDER_CLASSES.items():
        module = _load_adapter(provider)
        adapter: ProviderAdapter = getattr(module, class_name)(
            PLATFORM_ROOT / "iac" / provider / "platform-plan.yaml"
        )

        execution = adapter.deploy("development")

        assert execution.provider == provider
        assert execution.environment == "development"
        assert execution.dry_run is True
        assert execution.status == "planned"
        assert {action.name for action in execution.actions} == {
            "model-artifacts",
            "model-images",
            "model-serving",
            "prediction-logs",
            "model-registry",
        }
        assert all(action.status == "planned" for action in execution.actions)


def test_cloud_provider_adapters_do_not_import_sdks() -> None:
    forbidden = ("boto3", "google.cloud", "azure.identity", "azure.mgmt")
    for provider in PROVIDER_CLASSES:
        source = (PLATFORM_ROOT / "adapters" / provider / "adapter.py").read_text()
        assert not any(name in source for name in forbidden)


def _load_adapter(provider: str):
    adapter_path = PLATFORM_ROOT / "adapters" / provider / "adapter.py"
    spec = importlib.util.spec_from_file_location(f"{provider}_provider_adapter", adapter_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
