from pathlib import Path
import json
import shutil

from ml_production_ecosystem.production_patterns.provider_portability import validate_provider_portability

ROOT = Path(__file__).resolve().parents[1]

def test_provider_portability_passes_current_provider_plans(tmp_path: Path) -> None:
    report = validate_provider_portability(ROOT, tmp_path / "portability.json")

    assert report["status"] == "passed"
    assert report["providers"] == ["local", "aws", "gcp", "azure"]
    assert report["failures"] == []
    assert "model-artifacts" in report["resource_names_by_provider"]["aws"]
    assert "MODEL_REGISTRY_TOKEN" in report["secret_targets_by_provider"]["gcp"]
    assert json.loads((tmp_path / "portability.json").read_text()) == report

def test_provider_portability_fails_when_cloud_resource_sets_diverge(tmp_path: Path) -> None:
    source_root = ROOT / "configs" / "platform"
    target_root = tmp_path / "configs" / "platform"
    shutil.copytree(source_root, target_root)
    azure_plan = target_root / "iac" / "azure" / "platform-plan.yaml"
    azure_plan.write_text(azure_plan.read_text().replace("name: model-serving", "name: azure-only-serving", 1))

    report = validate_provider_portability(tmp_path, tmp_path / "portability.json")

    assert report["status"] == "failed"
    assert "cloud provider resource names differ" in report["failures"]
