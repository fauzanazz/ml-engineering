"""Local autoscaling decision simulation from load and SLO reports."""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = Path("artifacts/reports/scale-reliability/autoscaling-decision.json")
DEFAULT_TARGET_REQUESTS_PER_REPLICA = 50


def build_autoscaling_decision(
    load_report_path: Path,
    slo_report_path: Path | None = None,
    current_replicas: int = 1,
    min_replicas: int = 1,
    max_replicas: int = 4,
    target_requests_per_replica: int = DEFAULT_TARGET_REQUESTS_PER_REPLICA,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    _validate_bounds(current_replicas, min_replicas, max_replicas, target_requests_per_replica)
    load_report = _read_json(load_report_path)
    slo_report = _read_json(slo_report_path) if slo_report_path else {}

    request_count = int(load_report.get("request_count", 0))
    concurrency = int(load_report.get("concurrency", 1))
    error_count = int(load_report.get("error_count", 0))
    latency = load_report.get("latency_ms", {})
    latency_p95 = float(latency.get("p95", 0.0)) if isinstance(latency, dict) else 0.0
    breached_slos = [str(item) for item in slo_report.get("breached_slos", [])] if isinstance(slo_report.get("breached_slos", []), list) else []

    desired_by_load = _ceil_div(max(request_count, concurrency), target_requests_per_replica)
    desired_replicas = max(min_replicas, current_replicas, desired_by_load)
    reasons = []

    if breached_slos:
        desired_replicas = max(desired_replicas, current_replicas + 1)
        reasons.append(f"SLO breaches: {', '.join(breached_slos)}")
    if error_count > 0:
        desired_replicas = max(desired_replicas, current_replicas + 1)
        reasons.append(f"load errors: {error_count}")
    if desired_by_load > current_replicas:
        reasons.append(f"load target requires {desired_by_load} replicas")

    desired_replicas = min(max_replicas, desired_replicas)
    if desired_replicas > current_replicas:
        action = "scale_up"
    elif _can_scale_down(load_report, slo_report, current_replicas, min_replicas, target_requests_per_replica):
        action = "scale_down"
        desired_replicas = current_replicas - 1
        reasons.append("load and SLOs within budget below scale-down target")
    else:
        action = "hold"
        reasons.append("current replica count fits local simulation target")

    report: dict[str, Any] = {
        "status": "ready",
        "action": action,
        "current_replicas": current_replicas,
        "desired_replicas": desired_replicas,
        "min_replicas": min_replicas,
        "max_replicas": max_replicas,
        "target_requests_per_replica": target_requests_per_replica,
        "load": {
            "request_count": request_count,
            "concurrency": concurrency,
            "error_count": error_count,
            "latency_p95_ms": latency_p95,
        },
        "slo_status": str(slo_report.get("status", "not-provided")),
        "breached_slos": breached_slos,
        "reasons": reasons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _can_scale_down(
    load_report: dict[str, Any],
    slo_report: dict[str, Any],
    current_replicas: int,
    min_replicas: int,
    target_requests_per_replica: int,
) -> bool:
    if current_replicas <= min_replicas:
        return False
    if slo_report and slo_report.get("status") != "within_budget":
        return False
    request_count = int(load_report.get("request_count", 0))
    error_count = int(load_report.get("error_count", 0))
    return error_count == 0 and request_count < (current_replicas - 1) * target_requests_per_replica


def _validate_bounds(
    current_replicas: int,
    min_replicas: int,
    max_replicas: int,
    target_requests_per_replica: int,
) -> None:
    if min_replicas < 1:
        raise ValueError("min_replicas must be at least 1")
    if max_replicas < min_replicas:
        raise ValueError("max_replicas must be greater than or equal to min_replicas")
    if current_replicas < min_replicas or current_replicas > max_replicas:
        raise ValueError("current_replicas must be within min/max bounds")
    if target_requests_per_replica < 1:
        raise ValueError("target_requests_per_replica must be at least 1")


def _ceil_div(value: int, divisor: int) -> int:
    return max(1, (value + divisor - 1) // divisor)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate local autoscaling decision from load and SLO reports.")
    parser.add_argument("--load-report", type=Path, required=True)
    parser.add_argument("--slo-report", type=Path)
    parser.add_argument("--current-replicas", type=int, default=1)
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=4)
    parser.add_argument("--target-requests-per-replica", type=int, default=DEFAULT_TARGET_REQUESTS_PER_REPLICA)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_autoscaling_decision(
        load_report_path=args.load_report,
        slo_report_path=args.slo_report,
        current_replicas=args.current_replicas,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        target_requests_per_replica=args.target_requests_per_replica,
        output_path=args.output_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
