"""Machine-readable deployment demo checks for local serving."""

from pathlib import Path
import argparse
import json

from .monitoring_loop import HttpClient, evaluate_monitoring_summary

DEFAULT_OUTPUT_PATH = Path("artifacts/reports/production-patterns/deployment-demo.json")


def build_deployment_demo_report(
    base_url: str | None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_error_count: int = 0,
    max_drift_score: float = 0.2,
    max_latency_ms_last: float = 100.0,
    http_client: HttpClient | None = None,
) -> dict[str, object]:
    if base_url is None:
        report: dict[str, object] = {"status": "skipped", "reason": "base URL not provided", "checks": []}
    else:
        summary = evaluate_monitoring_summary(
            base_url=base_url,
            max_error_count=max_error_count,
            max_drift_score=max_drift_score,
            max_latency_ms_last=max_latency_ms_last,
            http_client=http_client,
        )
        report = {
            "status": "passed" if summary["status"] == "healthy" else "failed",
            "base_url": base_url,
            "checks": summary["checks"],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deployment demo checks and write JSON report.")
    parser.add_argument("--base-url")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-error-count", type=int, default=0)
    parser.add_argument("--max-drift-score", type=float, default=0.2)
    parser.add_argument("--max-latency-ms-last", type=float, default=100.0)
    args = parser.parse_args()

    report = build_deployment_demo_report(
        base_url=args.base_url,
        output_path=args.output_path,
        max_error_count=args.max_error_count,
        max_drift_score=args.max_drift_score,
        max_latency_ms_last=args.max_latency_ms_last,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
