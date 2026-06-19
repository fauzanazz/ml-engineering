"""Audit progress toward local-first/model-agnostic/provider-agnostic goal."""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(".")
DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/goal-readiness.json")

CHECKS = (
    {
        "name": "readme_vision",
        "description": "README states local-first, model-agnostic, provider-agnostic vision.",
        "evidence": ("README.md",),
        "required_text": ("local-first", "model-agnostic", "provider-agnostic"),
    },
    {
        "name": "local_lifecycle_without_cloud",
        "description": "Local lifecycle and deployment smoke scripts run without cloud credentials.",
        "evidence": ("scripts/smoke-local-lifecycle.sh", "scripts/smoke-local-deployment.sh"),
        "required_text": ("production-lifecycle-demo", "local lifecycle smoke passed", "local deployment smoke passed"),
    },
    {
        "name": "easy_path_documented",
        "description": "Add data through graph lifecycle is documented.",
        "evidence": ("docs/lifecycle-easy-path.md", "docs/local-lifecycle-runbook.md"),
        "required_text": ("add data", "train", "validate", "approve", "drift", "continual", "graph"),
    },
    {
        "name": "model_contracts_reusable",
        "description": "At least two model contract schema families exist.",
        "evidence": ("schemas/recommendation/input.json", "schemas/generic_classifier/input.json"),
        "required_text": (),
    },
    {
        "name": "generic_model_workflow",
        "description": "Command-trained non-recommender model has activation and rollback tests.",
        "evidence": ("tests/test_generic_classifier_command.py", "tests/test_rollback_generic_classifier.py"),
        "required_text": ("tiny-threshold-classifier", "rollback"),
    },
    {
        "name": "provider_boundaries_enforced",
        "description": "Provider boundaries, portability, swap matrix, and policy checks are tested.",
        "evidence": (
            "tests/test_provider_boundaries.py",
            "tests/test_provider_portability.py",
            "tests/test_provider_swap_matrix.py",
            "tests/test_policy_references.py",
        ),
        "required_text": ("provider", "policy", "portability"),
    },
    {
        "name": "cloud_provider_adapters",
        "description": "AWS/GCP/Azure adapters load and preview provider-neutral deployments without provider SDK dependencies.",
        "evidence": (
            "configs/platform/adapters/aws/adapter.py",
            "configs/platform/adapters/gcp/adapter.py",
            "configs/platform/adapters/azure/adapter.py",
            "tests/test_cloud_provider_adapters.py",
        ),
        "required_text": ("ProviderAdapter", "MODEL_REGISTRY_TOKEN", "deploy", "planned"),
    },
    {
        "name": "iac_secret_policy_as_code",
        "description": "Local/AWS/GCP/Azure IaC, secret refs, policies, and local injection manifest exist.",
        "evidence": (
            "configs/platform/iac/local/platform-plan.yaml",
            "configs/platform/iac/aws/platform-plan.yaml",
            "configs/platform/iac/gcp/platform-plan.yaml",
            "configs/platform/iac/azure/platform-plan.yaml",
            "configs/platform/policies/local/model-registry-read.yaml",
            "configs/platform/policies/aws/model-registry-read.yaml",
            "configs/platform/policies/gcp/model-registry-read.yaml",
            "configs/platform/policies/azure/model-registry-read.yaml",
            "configs/platform/secrets/local/secret-injections.yaml",
        ),
        "required_text": (),
    },
    {
        "name": "local_adapter_executable",
        "description": "Local adapter can create filesystem resources from plan references.",
        "evidence": ("configs/platform/adapters/local/adapter.py", "tests/test_local_platform.py", "pyproject.toml"),
        "required_text": ("ensure_resources", "production-apply-local-platform"),
    },
    {
        "name": "local_canary_workflow",
        "description": "Local canary decision uses deployment demo, drift, and approval evidence.",
        "evidence": ("src/ml_production_ecosystem/production_patterns/canary_decision.py", "tests/test_canary_decision.py", "scripts/smoke-local-deployment.sh"),
        "required_text": ("canary", "promote", "rollback"),
    },
    {
        "name": "local_canary_router",
        "description": "Local traffic-splitting canary simulation routes stable and candidate model requests.",
        "evidence": ("src/ml_production_ecosystem/production_patterns/canary_router.py", "tests/test_canary_router.py", "docs/domains/production-patterns/release-checklist.md", "pyproject.toml"),
        "required_text": ("production-canary-router", "local-traffic-splitting-simulation", "rollback"),
    },
    {
        "name": "slo_burn_rate_simulation",
        "description": "Local SLO burn-rate simulation exists for load and drift evidence.",
        "evidence": ("src/ml_production_ecosystem/scale_reliability/slo_burn_rate.py", "tests/test_slo_burn_rate.py", "docs/domains/scale-reliability/slo-definition.md"),
        "required_text": ("burn_rate", "scale-slo-burn-rate"),
    },
    {
        "name": "multi_window_burn_rate_alerting",
        "description": "Local multi-window burn-rate alert simulation exists.",
        "evidence": ("src/ml_production_ecosystem/scale_reliability/burn_rate_alert.py", "tests/test_burn_rate_alert.py", "docs/domains/scale-reliability/burn-rate-alerting.md"),
        "required_text": ("scale-burn-rate-alert", "critical", "warning"),
    },
    {
        "name": "autoscaling_decision_simulation",
        "description": "Local autoscaling decision simulation exists from load and SLO evidence.",
        "evidence": ("src/ml_production_ecosystem/scale_reliability/autoscaling_decision.py", "tests/test_autoscaling_decision.py", "docs/domains/scale-reliability/autoscaling-simulation.md"),
        "required_text": ("scale-autoscaling-decision", "scale_up", "scale_down"),
    },
    {
        "name": "distributed_load_aggregation",
        "description": "Local distributed-load shard aggregation exists.",
        "evidence": ("src/ml_production_ecosystem/scale_reliability/load_aggregate.py", "tests/test_load_aggregate.py", "docs/domains/scale-reliability/distributed-load-aggregation.md"),
        "required_text": ("scale-aggregate-load", "shard", "distributed-load"),
    },
    {
        "name": "local_cost_estimation",
        "description": "Local cost estimate exists from autoscaling and load evidence.",
        "evidence": ("src/ml_production_ecosystem/scale_reliability/cost_estimate.py", "tests/test_cost_estimate.py", "docs/domains/scale-reliability/cost-estimation.md"),
        "required_text": ("scale-cost-estimate", "estimated_monthly_cost", "learning-units"),
    },
    {
        "name": "local_kubernetes_parity",
        "description": "Local Kubernetes/kind manifest validation exists without applying cluster changes.",
        "evidence": ("configs/platform/local/kubernetes/foundation-api.yaml", "src/ml_production_ecosystem/production_patterns/local_kubernetes.py", "tests/test_local_kubernetes.py", "pyproject.toml"),
        "required_text": ("foundation-api", "secretKeyRef", "production-validate-local-kubernetes"),
    },
    {
        "name": "local_scheduler_plan",
        "description": "Local cron-compatible scheduler plan validation and runtime exist before managed schedulers.",
        "evidence": ("configs/platform/local/scheduler/jobs.yaml", "src/ml_production_ecosystem/production_patterns/local_scheduler.py", "tests/test_local_scheduler.py", "pyproject.toml"),
        "required_text": ("production-validate-local-scheduler", "production-run-local-scheduler", "local-scheduler-runtime", "cron-compatible"),
    },
)

KNOWN_GAPS: tuple[str, ...] = ()


def build_goal_readiness_report(
    root: Path = DEFAULT_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    checks = [_evaluate_check(root, check) for check in CHECKS]
    failed = [check["name"] for check in checks if check["status"] != "passed"]
    report: dict[str, Any] = {
        "status": "in_progress" if failed or KNOWN_GAPS else "ready",
        "passed_checks": [check["name"] for check in checks if check["status"] == "passed"],
        "failed_checks": failed,
        "checks": checks,
        "known_gaps": list(KNOWN_GAPS),
        "completion_claim": "not_complete" if KNOWN_GAPS or failed else "complete_candidate",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _evaluate_check(root: Path, check: dict[str, Any]) -> dict[str, Any]:
    missing_files = [path for path in check["evidence"] if not (root / path).exists()]
    missing_text = _missing_required_text(root, check)
    failures = []
    if missing_files:
        failures.append(f"missing evidence files: {', '.join(missing_files)}")
    if missing_text:
        failures.append(f"missing required text: {', '.join(missing_text)}")
    return {
        "name": check["name"],
        "description": check["description"],
        "status": "passed" if not failures else "failed",
        "evidence": list(check["evidence"]),
        "failures": failures,
    }


def _missing_required_text(root: Path, check: dict[str, Any]) -> list[str]:
    required_text = [str(text).lower() for text in check["required_text"]]
    if not required_text:
        return []
    text = "\n".join(
        (root / path).read_text(errors="ignore")
        for path in check["evidence"]
        if (root / path).exists()
    ).lower()
    return [required for required in required_text if required not in text]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit goal readiness evidence and known gaps.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_goal_readiness_report(args.root, args.output_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
