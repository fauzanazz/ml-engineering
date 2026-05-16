from pathlib import Path
import json

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_foundation_api_dockerfile_packages_local_service() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "FROM python:3.13-slim" in dockerfile
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "COPY templates ./templates" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert 'CMD ["foundation-serve-recommender", "--host", "0.0.0.0", "--port", "8000"]' in dockerfile


def test_monitoring_compose_services_are_defined() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = compose["services"]

    assert "foundation-api" in services
    foundation_api = services["foundation-api"]
    assert foundation_api["build"]["context"] == "."
    assert foundation_api["build"]["dockerfile"] == "Dockerfile"
    assert foundation_api["command"] == ["foundation-serve-recommender", "--host", "0.0.0.0", "--port", "8000"]
    assert "8000:8000" in foundation_api["ports"]
    assert "./01-foundation/artifacts:/app/01-foundation/artifacts" in foundation_api["volumes"]
    assert "./01-foundation/registry:/app/01-foundation/registry" in foundation_api["volumes"]
    assert "./01-foundation/logs:/app/01-foundation/logs" in foundation_api["volumes"]

    assert "prometheus" in services
    assert services["prometheus"]["image"] == "prom/prometheus:v2.55.1"
    assert "9090:9090" in services["prometheus"]["ports"]
    assert "./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" in services["prometheus"]["volumes"]

    assert "grafana" in services
    assert services["grafana"]["image"] == "grafana/grafana:11.3.0"
    assert "3000:3000" in services["grafana"]["ports"]
    assert "./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro" in services["grafana"]["volumes"]


def test_prometheus_scrapes_compose_api_metrics_endpoint() -> None:
    config = yaml.safe_load((ROOT / "monitoring" / "prometheus" / "prometheus.yml").read_text())

    scrape_configs = {item["job_name"]: item for item in config["scrape_configs"]}
    foundation_api = scrape_configs["foundation-recommender-api"]

    assert foundation_api["metrics_path"] == "/metrics"
    assert foundation_api["static_configs"] == [{"targets": ["foundation-api:8000"]}]


def test_grafana_prometheus_datasource_is_provisioned() -> None:
    datasource_path = ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
    config = yaml.safe_load(datasource_path.read_text())

    datasource = config["datasources"][0]
    assert datasource["name"] == "Prometheus"
    assert datasource["type"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"
    assert datasource["isDefault"] is True


def test_grafana_dashboard_contains_foundation_metrics() -> None:
    dashboard_path = ROOT / "monitoring" / "grafana" / "provisioning" / "dashboards" / "foundation-recommender.json"
    dashboard = json.loads(dashboard_path.read_text())
    panel_queries = "\n".join(
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    )

    assert dashboard["title"] == "Foundation Recommender Observability"
    assert "foundation_prediction_requests_total" in panel_queries
    assert "rate(foundation_prediction_requests_total" in panel_queries
    assert "foundation_prediction_errors_total" in panel_queries
    assert "rate(foundation_prediction_errors_total" in panel_queries
    assert "foundation_prediction_latency_ms_last" in panel_queries
    assert "foundation_prediction_latency_ms_sum / foundation_prediction_latency_ms_count" in panel_queries
    assert "foundation_prediction_requests_total" in panel_queries and "model_name" in panel_queries and "model_version" in panel_queries


def test_monitoring_run_flow_is_documented() -> None:
    docs = (ROOT / "docs" / "features" / "step-9-local-monitoring-stack.md").read_text()

    expected_steps = [
        "uv run foundation-train-recommender",
        "uv run foundation-set-active-model",
        "uv run foundation-serve-recommender --host 0.0.0.0 --port 8000",
        "docker compose up -d foundation-api prometheus grafana",
        "docker compose up -d prometheus grafana",
        "curl -X POST http://127.0.0.1:8000/predict/v1",
        "http://127.0.0.1:3000",
    ]

    for step in expected_steps:
        assert step in docs
