"""Local SLO burn-rate simulation from load and drift reports."""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = Path("artifacts/reports/scale-reliability/slo-burn-rate.json")
DEFAULT_AVAILABILITY_TARGET = 0.99
DEFAULT_ERROR_RATE_TARGET = 0.01
DEFAULT_LATENCY_P95_MS_TARGET = 200.0
DEFAULT_DRIFT_SCORE_TARGET = 0.2


def build_slo_burn_rate_report(
    load_report_path: Path,
    drift_report_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    availability_target: float = DEFAULT_AVAILABILITY_TARGET,
    error_rate_target: float = DEFAULT_ERROR_RATE_TARGET,
    latency_p95_ms_target: float = DEFAULT_LATENCY_P95_MS_TARGET,
    drift_score_target: float = DEFAULT_DRIFT_SCORE_TARGET,
) -> dict[str, Any]:
    load_report = _read_json(load_report_path)
    drift_report = _read_json(drift_report_path) if drift_report_path else {}

    request_count = int(load_report.get("request_count", 0))
    success_count = int(load_report.get("success_count", 0))
    error_count = int(load_report.get("error_count", 0))
    latency = load_report.get("latency_ms", {})
    latency_p95 = float(latency.get("p95", 0.0)) if isinstance(latency, dict) else 0.0
    observed_error_rate = (error_count / request_count) if request_count else 1.0
    observed_availability = (success_count / request_count) if request_count else 0.0
    drift_score = _drift_score(drift_report)

    checks = {
        "availability": _budget_check(
            observed_bad_rate=max(0.0, 1.0 - observed_availability),
            allowed_bad_rate=max(0.0, 1.0 - availability_target),
            observed_value=observed_availability,
            target=availability_target,
            comparison=">=",
        ),
        "error_rate": _budget_check(
            observed_bad_rate=observed_error_rate,
            allowed_bad_rate=error_rate_target,
            observed_value=observed_error_rate,
            target=error_rate_target,
            comparison="<=",
        ),
        "latency_p95_ms": _threshold_check(latency_p95, latency_p95_ms_target),
    }
    if drift_report:
        checks["drift_score"] = _threshold_check(drift_score, drift_score_target)

    breached = [name for name, check in checks.items() if check["status"] == "breached"]
    report: dict[str, Any] = {
        "status": "breached" if breached else "within_budget",
        "load_report_path": str(load_report_path),
        "drift_report_path": str(drift_report_path) if drift_report_path else None,
        "request_count": request_count,
        "checks": checks,
        "breached_slos": breached,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _budget_check(
    observed_bad_rate: float,
    allowed_bad_rate: float,
    observed_value: float,
    target: float,
    comparison: str,
) -> dict[str, Any]:
    burn_rate = float("inf") if allowed_bad_rate <= 0 and observed_bad_rate > 0 else observed_bad_rate / allowed_bad_rate
    return {
        "status": "breached" if burn_rate > 1.0 else "within_budget",
        "observed": round(observed_value, 6),
        "target": target,
        "comparison": comparison,
        "burn_rate": round(burn_rate, 6) if burn_rate != float("inf") else "inf",
    }


def _threshold_check(observed: float, target: float) -> dict[str, Any]:
    burn_rate = observed / target if target > 0 else float("inf")
    return {
        "status": "breached" if observed > target else "within_budget",
        "observed": round(observed, 6),
        "target": target,
        "comparison": "<=",
        "burn_rate": round(burn_rate, 6) if burn_rate != float("inf") else "inf",
    }


def _drift_score(report: dict[str, Any]) -> float:
    if "score" in report:
        return float(report["score"])
    payload = report.get("payload", {})
    if isinstance(payload, dict) and "drift_score" in payload:
        return float(payload["drift_score"])
    return 0.0


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate local SLO burn-rate checks.")
    parser.add_argument("--load-report", type=Path, required=True)
    parser.add_argument("--drift-report", type=Path)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--availability-target", type=float, default=DEFAULT_AVAILABILITY_TARGET)
    parser.add_argument("--error-rate-target", type=float, default=DEFAULT_ERROR_RATE_TARGET)
    parser.add_argument("--latency-p95-ms-target", type=float, default=DEFAULT_LATENCY_P95_MS_TARGET)
    parser.add_argument("--drift-score-target", type=float, default=DEFAULT_DRIFT_SCORE_TARGET)
    args = parser.parse_args()

    report = build_slo_burn_rate_report(
        load_report_path=args.load_report,
        drift_report_path=args.drift_report,
        output_path=args.output_path,
        availability_target=args.availability_target,
        error_rate_target=args.error_rate_target,
        latency_p95_ms_target=args.latency_p95_ms_target,
        drift_score_target=args.drift_score_target,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
