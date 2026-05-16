from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "configs" / "production-patterns" / "alerts" / "rules.yaml"
RUNBOOK_PATH = ROOT / "docs" / "domains" / "production-patterns" / "alerting-runbook.md"


def test_alert_rules_parse_with_required_fields() -> None:
    assert RULES_PATH.exists()
    config = yaml.safe_load(RULES_PATH.read_text())

    rules = config["rules"]
    rule_names = {rule["name"] for rule in rules}
    assert rule_names == {
        "high_error_count",
        "high_latency",
        "high_drift_score",
        "api_unhealthy",
    }

    for rule in rules:
        assert rule["condition"]
        assert "threshold" in rule
        assert rule["severity"] in {"warning", "critical"}
        assert rule["runbook"] == "docs/domains/production-patterns/alerting-runbook.md"


def test_alert_rules_tie_to_monitoring_metrics_and_health_endpoint() -> None:
    source = RULES_PATH.read_text()

    assert "prediction_error_count" in source
    assert "prediction_latency_ms_last" in source
    assert "drift_score" in source
    assert "/health" in source


def test_alerting_runbook_maps_each_alert_to_manual_actions() -> None:
    assert RUNBOOK_PATH.exists()
    runbook = RUNBOOK_PATH.read_text()

    for section in ["Symptoms", "Checks", "Immediate Action", "Escalation", "Recovery"]:
        assert f"## {section}" in runbook

    for rule_name in ["high_error_count", "high_latency", "high_drift_score", "api_unhealthy"]:
        assert rule_name in runbook

    assert "production-monitor" in runbook
    assert "GET /health" in runbook
    assert "GET /metrics.json" in runbook
    assert "GET /drift" in runbook
