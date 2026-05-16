"""Local cost estimate from load and autoscaling decision reports."""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = Path("artifacts/reports/scale-reliability/cost-estimate.json")
DEFAULT_REPLICA_HOURLY_COST = 0.05
DEFAULT_HOURS_PER_MONTH = 730


def build_cost_estimate(
    autoscaling_report_path: Path,
    load_report_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    replica_hourly_cost: float = DEFAULT_REPLICA_HOURLY_COST,
    hours_per_month: int = DEFAULT_HOURS_PER_MONTH,
) -> dict[str, Any]:
    if replica_hourly_cost < 0:
        raise ValueError("replica_hourly_cost must be non-negative")
    if hours_per_month < 1:
        raise ValueError("hours_per_month must be at least 1")

    autoscaling = _read_json(autoscaling_report_path)
    load = _read_json(load_report_path) if load_report_path else {}
    current_replicas = int(autoscaling.get("current_replicas", 0))
    desired_replicas = int(autoscaling.get("desired_replicas", current_replicas))
    delta_replicas = desired_replicas - current_replicas
    hourly_cost = desired_replicas * replica_hourly_cost
    monthly_cost = hourly_cost * hours_per_month
    request_count = int(load.get("request_count", autoscaling.get("load", {}).get("request_count", 0)))
    cost_per_1000_requests = (hourly_cost / request_count * 1000) if request_count else None

    report: dict[str, Any] = {
        "status": "estimated",
        "autoscaling_report_path": str(autoscaling_report_path),
        "load_report_path": str(load_report_path) if load_report_path else None,
        "replica_hourly_cost": replica_hourly_cost,
        "hours_per_month": hours_per_month,
        "current_replicas": current_replicas,
        "desired_replicas": desired_replicas,
        "delta_replicas": delta_replicas,
        "estimated_hourly_cost": round(hourly_cost, 6),
        "estimated_monthly_cost": round(monthly_cost, 6),
        "request_count_basis": request_count,
        "estimated_cost_per_1000_requests": round(cost_per_1000_requests, 6) if cost_per_1000_requests is not None else None,
        "currency": "learning-units",
        "boundary": "local estimate only; not cloud bill prediction",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate local simulated serving cost from scaling reports.")
    parser.add_argument("--autoscaling-report", type=Path, required=True)
    parser.add_argument("--load-report", type=Path)
    parser.add_argument("--replica-hourly-cost", type=float, default=DEFAULT_REPLICA_HOURLY_COST)
    parser.add_argument("--hours-per-month", type=int, default=DEFAULT_HOURS_PER_MONTH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    report = build_cost_estimate(
        autoscaling_report_path=args.autoscaling_report,
        load_report_path=args.load_report,
        output_path=args.output_path,
        replica_hourly_cost=args.replica_hourly_cost,
        hours_per_month=args.hours_per_month,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
