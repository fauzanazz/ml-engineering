"""Safe local model rollback entrypoint for production patterns."""

import argparse
import json
from pathlib import Path

from ml_production_ecosystem.shared.model_storage.registry import list_model_versions, set_active_model

DEFAULT_REPORT_PATH = Path("02-production-patterns/reports/rollback.json")


def _write_report(report_path: Path, summary: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))


def _failed_summary(model_name: str, target_version: str, error: str) -> dict[str, object]:
    return {
        "status": "failed",
        "model_name": model_name,
        "target_version": target_version,
        "error": error,
    }


def rollback_model(
    registry_path: Path,
    model_name: str,
    target_version: str,
    reason: str,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    versions = list_model_versions(registry_path, model_name)
    if not versions:
        return _failed_summary(model_name, target_version, "model not found")

    if not any(version.get("version") == target_version for version in versions):
        return _failed_summary(model_name, target_version, "target version not found")

    set_active_model(registry_path, model_name, target_version)
    summary = {
        "status": "rolled_back",
        "model_name": model_name,
        "target_version": target_version,
        "reason": reason,
        "report_path": str(report_path),
    }
    _write_report(report_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollback active model to a registered known-good version.")
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    summary = rollback_model(
        registry_path=args.registry_path,
        model_name=args.model_name,
        target_version=args.target_version,
        reason=args.reason,
        report_path=args.report_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
