"""One-command local lifecycle demo for model-agnostic production flow."""

from pathlib import Path
import argparse
import json
from typing import Any

import yaml

from shared.model_storage.registry import set_active_model
from .approval import build_approval_decision
from .continual_learning import build_continual_learning_decision
from .data_ingestion import build_dataset_manifest
from .deployment_demo import build_deployment_demo_report
from .drift_detection import build_drift_report
from .lifecycle_graph import DEFAULT_HTML_PATH, build_lifecycle_graph
from .model_contract_manifest import build_model_contract_manifest
from .offline_validation import build_offline_validation_report
from .platform_plan import DEFAULT_PLAN_PATH, validate_platform_plan
from .retraining import DEFAULT_MODEL_NAME, run_retraining

DEFAULT_OUTPUT_PATH = Path("02-production-patterns/reports/lifecycle-demo.json")
DEFAULT_GRAPH_PATH = Path("02-production-patterns/reports/lifecycle-demo.mmd")
DEFAULT_DATASET_MANIFEST_PATH = Path("02-production-patterns/reports/dataset-manifest.json")
DEFAULT_VALIDATION_REPORT_PATH = Path("02-production-patterns/reports/offline-validation.json")
DEFAULT_APPROVAL_PATH = Path("02-production-patterns/reports/approval-decision.json")
DEFAULT_DEPLOYMENT_DEMO_PATH = Path("02-production-patterns/reports/deployment-demo.json")
DEFAULT_DRIFT_REPORT_PATH = Path("02-production-patterns/reports/drift-report.json")
DEFAULT_CONTINUAL_LEARNING_PATH = Path("02-production-patterns/reports/continual-learning-decision.json")
DEFAULT_MODEL_CONTRACT_PATH = Path("02-production-patterns/reports/model-contract-manifest.json")
DEFAULT_PLATFORM_REPORT_PATH = Path("02-production-patterns/reports/platform-plan-validation.json")


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as file:
        config = yaml.safe_load(file)
    if isinstance(config, dict):
        return config
    return {}


def _registry_path(config: dict[str, Any]) -> Path | None:
    registry = config.get("registry", {})
    if not isinstance(registry, dict):
        return None
    path = registry.get("path")
    if path is None:
        return None
    return Path(str(path))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_lifecycle_demo(
    config_path: Path,
    approve: bool = False,
    set_active: bool = False,
    base_url: str | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    graph_path: Path = DEFAULT_GRAPH_PATH,
    graph_html_path: Path = DEFAULT_HTML_PATH,
    model_contract_path: Path = DEFAULT_MODEL_CONTRACT_PATH,
    platform_plan_path: Path = DEFAULT_PLAN_PATH,
    platform_report_path: Path = DEFAULT_PLATFORM_REPORT_PATH,
    dataset_manifest_path: Path = DEFAULT_DATASET_MANIFEST_PATH,
    validation_report_path: Path = DEFAULT_VALIDATION_REPORT_PATH,
    approval_path: Path = DEFAULT_APPROVAL_PATH,
    deployment_demo_path: Path = DEFAULT_DEPLOYMENT_DEMO_PATH,
    drift_report_path: Path = DEFAULT_DRIFT_REPORT_PATH,
    continual_learning_path: Path = DEFAULT_CONTINUAL_LEARNING_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    max_error_count: int = 0,
    max_drift_score: float = 0.2,
    max_latency_ms_last: float = 100.0,
) -> dict[str, object]:
    config = _load_config(config_path)
    platform = validate_platform_plan(platform_plan_path, platform_report_path)
    model_contract = build_model_contract_manifest(config_path, model_contract_path)
    dataset = build_dataset_manifest(config_path, dataset_manifest_path)
    retraining = run_retraining(config_path=config_path, set_active=False, require_quality_gate=True, model_name=model_name)
    validation = build_offline_validation_report(config_path, validation_report_path)
    approval = build_approval_decision(validation_report_path, approved=approve, output_path=approval_path)

    activated = False
    if set_active and bool(approval["approved"]):
        registry_path = _registry_path(config)
        if registry_path is None:
            raise ValueError("registry.path is required when --set-active")
        set_active_model(registry_path, model_name, str(retraining["version"]))
        activated = True

    demo = build_deployment_demo_report(
        base_url=base_url,
        output_path=deployment_demo_path,
        max_error_count=max_error_count,
        max_drift_score=max_drift_score,
        max_latency_ms_last=max_latency_ms_last,
    )
    drift = build_drift_report(base_url=base_url, output_path=drift_report_path, threshold=max_drift_score)
    continual_learning = build_continual_learning_decision(drift_report_path, deployment_demo_path, continual_learning_path)
    graph = build_lifecycle_graph(graph_path, graph_html_path)

    summary = {
        "status": "completed",
        "platform": platform,
        "dataset": dataset,
        "model_contract": model_contract,
        "training": {
            "status": retraining["status"],
            "model_name": retraining["model_name"],
            "version": retraining["version"],
            "artifact_uri": retraining["artifact_uri"],
            "metrics_uri": retraining["metrics_uri"],
        },
        "offline_validation": validation,
        "approval": approval,
        "activation": {"set_active": activated},
        "deployment_demo": demo,
        "drift": drift,
        "continual_learning": continual_learning,
        "graph": graph,
    }
    _write_json(output_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local-first ML lifecycle demo.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--set-active", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--graph-path", type=Path)
    parser.add_argument("--graph-html-path", type=Path)
    parser.add_argument("--model-contract-path", type=Path)
    parser.add_argument("--platform-plan-path", type=Path)
    parser.add_argument("--platform-report-path", type=Path)
    parser.add_argument("--dataset-manifest-path", type=Path)
    parser.add_argument("--validation-report-path", type=Path)
    parser.add_argument("--approval-path", type=Path)
    parser.add_argument("--deployment-demo-path", type=Path)
    parser.add_argument("--drift-report-path", type=Path)
    parser.add_argument("--continual-learning-path", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--max-error-count", type=int)
    parser.add_argument("--max-drift-score", type=float)
    parser.add_argument("--max-latency-ms-last", type=float)
    args = parser.parse_args()

    summary = run_lifecycle_demo(
        config_path=args.config,
        approve=args.approve,
        set_active=args.set_active,
        base_url=args.base_url,
        output_path=args.output_path or DEFAULT_OUTPUT_PATH,
        graph_path=args.graph_path or DEFAULT_GRAPH_PATH,
        graph_html_path=args.graph_html_path or DEFAULT_HTML_PATH,
        model_contract_path=args.model_contract_path or DEFAULT_MODEL_CONTRACT_PATH,
        platform_plan_path=args.platform_plan_path or DEFAULT_PLAN_PATH,
        platform_report_path=args.platform_report_path or DEFAULT_PLATFORM_REPORT_PATH,
        dataset_manifest_path=args.dataset_manifest_path or DEFAULT_DATASET_MANIFEST_PATH,
        validation_report_path=args.validation_report_path or DEFAULT_VALIDATION_REPORT_PATH,
        approval_path=args.approval_path or DEFAULT_APPROVAL_PATH,
        deployment_demo_path=args.deployment_demo_path or DEFAULT_DEPLOYMENT_DEMO_PATH,
        drift_report_path=args.drift_report_path or DEFAULT_DRIFT_REPORT_PATH,
        continual_learning_path=args.continual_learning_path or DEFAULT_CONTINUAL_LEARNING_PATH,
        model_name=args.model_name or DEFAULT_MODEL_NAME,
        max_error_count=args.max_error_count if args.max_error_count is not None else 0,
        max_drift_score=args.max_drift_score if args.max_drift_score is not None else 0.2,
        max_latency_ms_last=args.max_latency_ms_last if args.max_latency_ms_last is not None else 100.0,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
