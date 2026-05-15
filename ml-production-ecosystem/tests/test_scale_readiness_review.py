from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "03-scale-and-reliability" / "docs" / "scale-readiness-review.md"


def test_scale_readiness_review_doc_exists() -> None:
    assert REVIEW_PATH.exists()


def test_scale_readiness_review_has_required_sections() -> None:
    doc = REVIEW_PATH.read_text()

    for section in [
        "## Completed Capabilities",
        "## Local Scale Readiness",
        "## Reliability Coverage",
        "## Real Million Request Gaps",
        "## Out Of Scope For 03",
        "## Next Module Decision",
        "## Step 39 Options",
    ]:
        assert section in doc


def test_scale_readiness_review_lists_steps_28_to_37() -> None:
    doc = REVIEW_PATH.read_text()

    for step in range(28, 38):
        assert f"Step {step}" in doc


def test_scale_readiness_review_lists_local_scale_capabilities() -> None:
    doc = REVIEW_PATH.read_text()

    for capability in [
        "repeated local API requests",
        "sequential and concurrent inference testing",
        "timeout and retry simulation",
        "controlled overload rejection with backpressure",
        "batch throughput baseline",
        "cache hit/miss measurement",
        "simple SLO targets",
        "failure injection validation",
        "reliability triage runbook",
    ]:
        assert capability in doc


def test_scale_readiness_review_lists_real_million_request_gaps() -> None:
    doc = REVIEW_PATH.read_text()

    for gap in [
        "real distributed load testing",
        "real horizontal autoscaling",
        "Kubernetes production deployment",
        "cloud load balancer",
        "multi-instance model serving",
        "distributed cache",
        "queue-based asynchronous inference",
        "batching optimization at high throughput",
        "model server optimization",
        "feature store at scale",
        "centralized logs/metrics/traces",
        "Multi-window SLO burn-rate alert simulation",
        "capacity planning with real traffic",
        "multi-region reliability",
        "cloud IAM/secrets/networking",
        "real cloud cost modeling",
    ]:
        assert gap in doc


def test_scale_readiness_review_states_boundary_and_next_decision() -> None:
    doc = REVIEW_PATH.read_text()

    for required in [
        "03 is local scale/reliability foundation, not real million-request foundation",
        "Move to 04-platform-and-cloud",
        "Go deeper inside 03-scale-and-reliability",
        "Pause implementation and manually review local load/reliability reports",
    ]:
        assert required in doc
