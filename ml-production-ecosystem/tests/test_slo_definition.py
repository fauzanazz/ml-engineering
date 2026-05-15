from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLO_DOC_PATH = ROOT / "03-scale-and-reliability" / "docs" / "slo-definition.md"


def test_slo_definition_doc_exists() -> None:
    assert SLO_DOC_PATH.exists()


def test_slo_definition_includes_required_slos() -> None:
    doc = SLO_DOC_PATH.read_text()

    for required in [
        "Availability",
        "Latency p95",
        "Error Rate",
        "Drift Threshold",
        ">= 99%",
        "<= 200 ms",
        "<= 1%",
        "<= 0.2",
    ]:
        assert required in doc


def test_slo_definition_maps_alerts_to_breaches() -> None:
    doc = SLO_DOC_PATH.read_text()

    for required in [
        "service health alert maps to availability SLO breach",
        "high latency alert maps to latency p95 SLO breach",
        "error count or error rate alert maps to error rate SLO breach",
        "drift alert maps to drift threshold SLO breach",
    ]:
        assert required in doc


def test_slo_definition_explains_learning_boundary() -> None:
    doc = SLO_DOC_PATH.read_text()

    for required in [
        "SLO is target behavior, not a guarantee",
        "SLI is measurement signal",
        "alert is operational notification",
        "learning defaults",
        "thresholds should be tuned with real traffic later",
        "drift threshold is model-quality risk signal",
        "scale-slo-burn-rate",
    ]:
        assert required in doc
