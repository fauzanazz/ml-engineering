from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "02-production-patterns" / "deploy" / "deployment-manifest.yaml"
DOC_PATH = ROOT / "02-production-patterns" / "docs" / "deployment-manifest.md"
COMPOSE_PATH = ROOT / "docker-compose.production.yaml"
COMPOSE_DOC_PATH = ROOT / "02-production-patterns" / "docs" / "production-compose.md"


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


def test_production_compose_matches_deployment_manifest_service() -> None:
    assert COMPOSE_PATH.exists()
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())

    service_name = manifest["service_name"]
    service = compose["services"][service_name]

    assert service_name == "foundation-api"
    assert service["image"] == manifest["image"]
    assert service["command"] == manifest["command"]
    assert service["ports"] == ["8000:8000"]
    assert manifest["health_endpoint"] in " ".join(service["healthcheck"]["test"])
    assert "01-foundation/artifacts" in " ".join(service["volumes"])
    assert "01-foundation/registry" in " ".join(service["volumes"])


def test_production_compose_doc_explains_start_verify_stop_flow() -> None:
    assert COMPOSE_DOC_PATH.exists()
    doc = COMPOSE_DOC_PATH.read_text()

    assert "docker compose -f docker-compose.production.yaml up --build foundation-api" in doc
    assert "uv run production-monitor" in doc
    assert "--base-url http://127.0.0.1:8000" in doc
    assert "--max-error-count 0" in doc
    assert "--max-drift-score 0.2" in doc
    assert "--max-latency-ms-last 100" in doc
    assert "docker compose -f docker-compose.production.yaml down" in doc
    assert "/health" in doc
    assert "/metrics" in doc
    assert "/drift" in doc
