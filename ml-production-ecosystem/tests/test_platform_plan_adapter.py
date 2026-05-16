from pathlib import Path

import pytest

from ml_production_ecosystem.shared.platform import PlatformPlanAdapter, ProviderAdapter

ROOT = Path(__file__).resolve().parents[1]

def test_platform_plan_adapter_loads_all_provider_plans() -> None:
    for provider in ("local", "aws", "gcp", "azure"):
        adapter: ProviderAdapter = PlatformPlanAdapter(
            ROOT / "configs" / "platform" / provider / "platform-plan.yaml"
        )

        plan = adapter.plan("development")

        assert plan.provider == provider
        assert {resource.name for resource in plan.resources} >= {"model-artifacts", "model-registry"}
        assert all(resource.provider == provider for resource in plan.resources)
        assert plan.secrets[0].provider == provider
        assert "token" in plan.secrets[0].name

def test_platform_plan_adapter_deploys_without_provider_sdk() -> None:
    adapter: ProviderAdapter = PlatformPlanAdapter(
        ROOT / "configs" / "platform" / "aws" / "platform-plan.yaml"
    )

    execution = adapter.deploy("development")

    assert execution.provider == "aws"
    assert execution.status == "planned"
    assert execution.dry_run is True
    assert execution.actions[0].status == "planned"
    assert "s3://" in execution.actions[0].uri

def test_platform_plan_adapter_rejects_environment_mismatch() -> None:
    adapter = PlatformPlanAdapter(ROOT / "configs" / "platform" / "local" / "platform-plan.yaml")

    with pytest.raises(ValueError, match="does not match"):
        adapter.plan("production")
