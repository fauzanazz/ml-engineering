from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "02-production-patterns" / "deploy" / "deployment-manifest.yaml"
DOC_PATH = ROOT / "02-production-patterns" / "docs" / "deployment-manifest.md"


def test_deployment_manifest_yaml_exists_and_parses() -> None:
    assert MANIFEST_PATH.exists()
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())

    assert manifest["service_name"] == "foundation-api"
    assert manifest["image"] == "ml-production-ecosystem-foundation-api"
    assert manifest["command"] == "uv run foundation-serve-recommender"
    assert manifest["port"] == 8000
    assert manifest["health_endpoint"] == "/health"
    assert manifest["metrics_endpoint"] == "/metrics"
    assert manifest["metrics_json_endpoint"] == "/metrics.json"
    assert manifest["drift_endpoint"] == "/drift"
    assert manifest["registry_path"] == "01-foundation/registry/models.json"
    assert manifest["release_checklist"] == "02-production-patterns/docs/release-checklist.md"
    assert "uv run production-rollback-model" in manifest["rollback_command"]


def test_deployment_manifest_doc_explains_fields_and_update_timing() -> None:
    assert DOC_PATH.exists()
    doc = DOC_PATH.read_text()

    for field in [
        "service_name",
        "image",
        "command",
        "port",
        "health_endpoint",
        "metrics_endpoint",
        "metrics_json_endpoint",
        "drift_endpoint",
        "registry_path",
        "release_checklist",
        "rollback_command",
    ]:
        assert field in doc

    assert "When to update" in doc
    assert "deployment-manifest.yaml" in doc
    assert "release-checklist.md" in doc
