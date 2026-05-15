from pathlib import Path
import importlib.util
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
PLATFORM_ROOT = ROOT / "04-platform-and-cloud"


def test_platform_stage_documents_provider_boundaries() -> None:
    readme = (PLATFORM_ROOT / "README.md").read_text()
    boundaries = (PLATFORM_ROOT / "docs" / "provider-boundaries.md").read_text()
    provider_swap = (PLATFORM_ROOT / "docs" / "provider-swap.md").read_text()

    assert "model-agnostic core workflows" in readme
    assert "Provider examples are adapters" in readme
    assert "shared/lifecycle" in readme
    assert "must not import AWS, GCP, Azure" in boundaries
    assert "LifecycleRun" in boundaries
    assert "local workflow still run without provider credentials" in boundaries
    assert "Provider swap changes only these files" in provider_swap
    assert "production-provider-swap-matrix" in provider_swap


def test_provider_adapter_and_iac_scopes_exist() -> None:
    for provider in ("local", "aws", "gcp", "azure"):
        adapter_readme = PLATFORM_ROOT / "adapters" / provider / "README.md"
        iac_readme = PLATFORM_ROOT / "iac" / provider / "README.md"

        assert adapter_readme.exists()
        assert iac_readme.exists()
        assert "shared contracts" in adapter_readme.read_text()
        assert "secret values" in iac_readme.read_text()


def test_secret_management_doc_forbids_secret_values() -> None:
    doc = (PLATFORM_ROOT / "docs" / "secret-management.md").read_text()

    assert "never commits secret values" in doc
    assert "Secret names" in doc
    assert "API keys, tokens, passwords" in doc


def test_local_provider_adapter_returns_reference_plan_only() -> None:
    adapter_module = load_local_adapter_module()
    config = adapter_module.LocalAdapterConfig(project_root=ROOT)
    adapter = adapter_module.LocalProviderAdapter(config)

    plan = adapter.plan(environment="development")

    assert plan.provider == "local"
    assert {resource.name for resource in plan.resources} == {
        "model-artifacts",
        "model-registry",
        "prediction-logs",
    }
    assert all(resource.provider == "local" for resource in plan.resources)
    assert all(resource.kind == "local-path" for resource in plan.resources)
    assert plan.secrets[0].name == "development/local/model-registry-token"
    assert plan.secrets[0].injection_target == "LOCAL_MODEL_REGISTRY_TOKEN"
    assert "secret-value" not in repr(plan)




def test_local_provider_adapter_can_apply_filesystem_resources(tmp_path: Path) -> None:
    adapter_module = load_local_adapter_module()
    config = adapter_module.LocalAdapterConfig(project_root=tmp_path)
    adapter = adapter_module.LocalProviderAdapter(config)

    summary = adapter.ensure_resources(environment="development")

    assert summary["status"] == "ready"
    assert (tmp_path / "01-foundation" / "artifacts").is_dir()
    assert (tmp_path / "01-foundation" / "logs").is_dir()
    assert (tmp_path / "01-foundation" / "registry").is_dir()
    assert summary["secret_references"] == ["LOCAL_MODEL_REGISTRY_TOKEN"]

def test_local_iac_plan_matches_adapter_contract() -> None:
    plan = yaml.safe_load((PLATFORM_ROOT / "iac" / "local" / "platform-plan.yaml").read_text())

    assert plan["provider"] == "local"
    assert plan["environment"] == "development"
    assert {resource["name"] for resource in plan["resources"]} == {
        "model-artifacts",
        "model-registry",
        "prediction-logs",
    }
    assert next(resource["uri"] for resource in plan["resources"] if resource["name"] == "model-artifacts") == "01-foundation/artifacts"
    assert plan["secrets"] == [
        {
            "provider": "local",
            "name": "development/local/model-registry-token",
            "injection_target": "LOCAL_MODEL_REGISTRY_TOKEN",
            "policy_ref": "local-env-file-reference",
        }
    ]


def test_cloud_iac_plans_share_provider_neutral_contract() -> None:
    required_resources = {
        "model-artifacts",
        "model-images",
        "model-serving",
        "prediction-logs",
        "model-registry",
    }

    for provider in ("aws", "gcp", "azure"):
        plan = yaml.safe_load((PLATFORM_ROOT / "iac" / provider / "platform-plan.yaml").read_text())

        assert plan["provider"] == provider
        assert plan["environment"] == "development"
        assert {resource["name"] for resource in plan["resources"]} == required_resources
        assert all(resource["kind"] for resource in plan["resources"])
        assert all(resource["uri"] for resource in plan["resources"])
        assert plan["secrets"] == [
            {
                "provider": provider,
                "name": f"development/{provider}/model-registry-token",
                "injection_target": "MODEL_REGISTRY_TOKEN",
                "policy_ref": f"policy/{provider}-model-registry-read",
            }
        ]


def test_core_code_does_not_import_cloud_providers() -> None:
    forbidden_import = re.compile(
        r"^\s*(?:from|import)\s+(boto3|botocore|google\.cloud|azure)(?:\b|\.)",
        re.MULTILINE,
    )
    allowed_root = ROOT / "04-platform-and-cloud"
    source_files = [
        path
        for path in ROOT.rglob("*.py")
        if ".venv" not in path.parts
        and "__pycache__" not in path.parts
        and allowed_root not in path.parents
    ]

    violations = [
        str(path.relative_to(ROOT))
        for path in source_files
        if forbidden_import.search(path.read_text())
    ]

    assert violations == []


def load_local_adapter_module():
    adapter_path = PLATFORM_ROOT / "adapters" / "local" / "adapter.py"
    spec = importlib.util.spec_from_file_location("local_platform_adapter", adapter_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
