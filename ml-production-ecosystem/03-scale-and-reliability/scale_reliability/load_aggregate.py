"""Aggregate local load-test shard reports into one distributed-load summary."""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = Path("03-scale-and-reliability/reports/distributed-load-aggregate.json")


def aggregate_load_reports(
    report_paths: list[Path],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("at least one report path is required")

    reports = [_read_json(path) for path in report_paths]
    request_count = sum(int(report.get("request_count", 0)) for report in reports)
    success_count = sum(int(report.get("success_count", 0)) for report in reports)
    error_count = sum(int(report.get("error_count", 0)) for report in reports)
    attempt_count = sum(int(report.get("attempt_count", 0)) for report in reports)
    retry_attempt_count = sum(int(report.get("retry_attempt_count", 0)) for report in reports)
    retried_request_count = sum(int(report.get("retried_request_count", 0)) for report in reports)
    retry_success_count = sum(int(report.get("retry_success_count", 0)) for report in reports)
    retry_exhausted_count = sum(int(report.get("retry_exhausted_count", 0)) for report in reports)
    latency_ms = _aggregate_latency(reports)
    errors = _aggregate_errors(reports)

    summary: dict[str, Any] = {
        "status": _status(success_count, error_count),
        "shard_count": len(reports),
        "report_paths": [str(path) for path in report_paths],
        "request_count": request_count,
        "success_count": success_count,
        "error_count": error_count,
        "attempt_count": attempt_count,
        "retry_attempt_count": retry_attempt_count,
        "retried_request_count": retried_request_count,
        "retry_success_count": retry_success_count,
        "retry_exhausted_count": retry_exhausted_count,
        "error_rate": round(error_count / request_count, 6) if request_count else 1.0,
        "latency_ms": latency_ms,
        "errors": errors,
        "shards": [_shard_summary(path, report) for path, report in zip(report_paths, reports, strict=True)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _aggregate_latency(reports: list[dict[str, Any]]) -> dict[str, float]:
    latencies = [report.get("latency_ms", {}) for report in reports]
    valid = [latency for latency in latencies if isinstance(latency, dict)]
    if not valid:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": min(float(latency.get("min", 0.0)) for latency in valid),
        "p50": max(float(latency.get("p50", 0.0)) for latency in valid),
        "p95": max(float(latency.get("p95", 0.0)) for latency in valid),
        "max": max(float(latency.get("max", 0.0)) for latency in valid),
    }


def _aggregate_errors(reports: list[dict[str, Any]]) -> dict[str, int]:
    errors: dict[str, int] = {}
    for report in reports:
        report_errors = report.get("errors", {})
        if not isinstance(report_errors, dict):
            continue
        for name, count in report_errors.items():
            errors[str(name)] = errors.get(str(name), 0) + int(count)
    return errors


def _shard_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "status": str(report.get("status", "unknown")),
        "request_count": int(report.get("request_count", 0)),
        "success_count": int(report.get("success_count", 0)),
        "error_count": int(report.get("error_count", 0)),
    }


def _status(success_count: int, error_count: int) -> str:
    if success_count == 0:
        return "failed"
    if error_count > 0:
        return "degraded"
    return "passed"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate local load-test shard reports.")
    parser.add_argument("--report", dest="reports", type=Path, action="append", required=True)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    summary = aggregate_load_reports(args.reports, args.output_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
