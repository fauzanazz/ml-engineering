from pathlib import Path
import json

from ml_production_ecosystem.production_patterns.platform_plan import validate_platform_plan

ROOT = Path(__file__).resolve().parents[1]


def test_validate_platform_plan_passes_reference_only_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "platform-plan.yaml"
    output_path = tmp_path / "platform-report.json"
    plan_path.write_text(
        """
provider: local
environment: development
resources:
  - kind: local-path
    name: model-artifacts
    uri: 01-foundation/artifacts
secrets:
  - provider: local
    name: development/local/model-registry-token
    injection_target: LOCAL_MODEL_REGISTRY_TOKEN
    policy_ref: local-env-file-reference
""".strip()
    )

    report = validate_platform_plan(plan_path, output_path)

    assert report["status"] == "passed"
    assert report["resource_count"] == 1
    assert report["secret_reference_count"] == 1
    assert report["failures"] == []
    assert json.loads(output_path.read_text()) == report


def test_validate_platform_plan_rejects_secret_values(tmp_path: Path) -> None:
    plan_path = tmp_path / "platform-plan.yaml"
    plan_path.write_text(
        """
provider: local
environment: development
resources: []
secrets:
  - provider: local
    name: demo
    injection_target: DEMO_SECRET
    policy_ref: local-env-file-reference
    value: do-not-commit
""".strip()
    )

    report = validate_platform_plan(plan_path, tmp_path / "platform-report.json")

    assert report["status"] == "failed"
    assert report["failures"] == ["secrets[0] contains forbidden value key value"]

def test_validate_all_provider_platform_plans_are_reference_only(tmp_path: Path) -> None:
    for provider in ("local", "aws", "gcp", "azure"):
        plan_path = ROOT / "04-platform-and-cloud" / "iac" / provider / "platform-plan.yaml"
        report = validate_platform_plan(plan_path, tmp_path / f"{provider}-report.json")

        assert report["status"] == "passed"
        assert report["provider"] == provider
        assert report["secret_reference_count"] == 1
        assert report["failures"] == []
