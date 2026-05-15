from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPE_REVIEW_PATH = ROOT / "02-production-patterns" / "docs" / "scope-review.md"


def test_scope_review_doc_has_required_sections() -> None:
    assert SCOPE_REVIEW_PATH.exists()
    doc = SCOPE_REVIEW_PATH.read_text()

    for section in [
        "Completed Capabilities",
        "End-to-End Flow",
        "Functional Requirements Coverage",
        "Known Gaps",
        "Out Of Scope For 02",
        "Recommended Next Module",
        "Step 28 Options",
    ]:
        assert f"## {section}" in doc


def test_scope_review_lists_steps_11_through_26() -> None:
    doc = SCOPE_REVIEW_PATH.read_text()

    for step in range(11, 27):
        assert f"Step {step}" in doc

    for capability in [
        "batch inference",
        "production-retrain",
        "quality gate",
        "production-monitor",
        "production-scheduled-retrain",
        "Airflow",
        "alert rules",
        "production-rollback-model",
        "release checklist",
        "deployment manifest",
        "local CI",
        "GitHub Actions",
        "production compose",
        "smoke test",
        "production-release-summary",
    ]:
        assert capability in doc


def test_scope_review_maps_functional_requirements_and_boundaries() -> None:
    doc = SCOPE_REVIEW_PATH.read_text()

    for requirement in [
        "config-driven training",
        "experiment/model tracking",
        "one-command deploy-ish path",
        "monitoring latency/drift",
        "scheduled retraining",
        "release/rollback evidence",
    ]:
        assert requirement in doc

    assert "production-pattern foundation, not scale foundation" in doc

    for gap in [
        "million request inference",
        "batching optimization at scale",
        "autoscaling",
        "distributed training",
        "feature store",
        "online A/B testing",
        "Kubernetes",
        "cloud IAM/secrets",
        "production SLO burn rates",
    ]:
        assert gap in doc

    assert "03-scale-and-reliability" in doc
