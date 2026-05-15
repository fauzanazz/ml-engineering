from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_PATH = ROOT / "03-scale-and-reliability" / "docs" / "reliability-runbook.md"


def test_reliability_runbook_doc_exists() -> None:
    assert RUNBOOK_PATH.exists()


def test_reliability_runbook_has_required_sections() -> None:
    doc = RUNBOOK_PATH.read_text()

    for section in [
        "## Overview",
        "## Signals",
        "## Overload",
        "## High Latency",
        "## High Error Rate",
        "## Stale Model",
        "## Drift",
        "## Rollback Decision",
        "## Links To 02 Production Patterns",
    ]:
        assert section in doc


def test_reliability_runbook_references_02_capabilities() -> None:
    doc = RUNBOOK_PATH.read_text()

    for required in [
        "production-monitor",
        "production-rollback-model",
        "release checklist",
        "alerting runbook",
        "live smoke test",
        "release summary",
        "scheduled retraining",
        "./scripts/smoke-test-foundation-api.sh http://127.0.0.1:8000",
    ]:
        assert required in doc


def test_reliability_runbook_explains_response_boundaries() -> None:
    doc = RUNBOOK_PATH.read_text()

    for required in [
        "runbook is local learning guide",
        "monitoring detects symptoms",
        "rollback mitigates bad release",
        "retraining mitigates stale/drift model behavior",
        "load behavior tools identify pressure points",
        "not every reliability issue needs rollback",
        "scale fixes come after clear diagnosis",
    ]:
        assert required in doc
