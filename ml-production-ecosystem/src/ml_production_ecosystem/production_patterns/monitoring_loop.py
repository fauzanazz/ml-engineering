"""Monitoring loop summary for production serving endpoints."""

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

import httpx


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> dict[str, Any]: ...


class HttpClient(Protocol):
    def get(self, url: str, timeout: float) -> HttpResponse: ...


def _failed_check(name: str, message: str) -> dict[str, object]:
    return {"name": name, "passed": False, "message": message}


def _passed_check(name: str, message: str) -> dict[str, object]:
    return {"name": name, "passed": True, "message": message}


def _get_json(base_url: str, path: str, http_client: HttpClient, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = http_client.get(url, timeout=timeout)
        if response.status_code >= 400:
            return None, f"HTTP {response.status_code}"
        return response.json(), None
    except Exception as error:
        return None, str(error)


def _threshold_check(name: str, actual: float, maximum: float) -> dict[str, object]:
    message = f"{actual} <= {maximum}"
    if actual <= maximum:
        return _passed_check(name, message)
    return _failed_check(name, message)


def run_monitoring_loop(
    metrics_path: Path,
    output_path: Path,
    maximums: dict[str, float],
) -> dict[str, object]:
    metrics = json.loads(metrics_path.read_text())
    checks = []
    for metric_name, maximum in maximums.items():
        actual = float(metrics.get(metric_name, 0.0))
        passed = actual <= maximum
        if passed:
            message = f"{metric_name} {actual} within maximum {float(maximum)}"
        else:
            message = f"{metric_name} {actual} above maximum {float(maximum)}"
        checks.append(
            {
                "check_name": f"{metric_name}_maximum",
                "passed": passed,
                "message": message,
            }
        )

    summary = {
        "status": "passed" if all(bool(check["passed"]) for check in checks) else "failed",
        "metrics_path": str(metrics_path),
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def evaluate_monitoring_summary(
    base_url: str,
    max_error_count: int,
    max_drift_score: float,
    max_latency_ms_last: float,
    http_client: HttpClient | None = None,
    timeout: float = 5.0,
) -> dict[str, object]:
    client = http_client or httpx.Client()
    health, health_error = _get_json(base_url, "/health", client, timeout)
    metrics, metrics_error = _get_json(base_url, "/metrics.json", client, timeout)
    drift, drift_error = _get_json(base_url, "/drift", client, timeout)

    checks: list[dict[str, object]] = []
    if health_error is not None:
        checks.append(_failed_check("health", health_error))
    elif health and health.get("status") == "ok":
        checks.append(_passed_check("health", "ok"))
    else:
        checks.append(_failed_check("health", str(health)))

    if metrics_error is not None or metrics is None:
        checks.append(_failed_check("error_count", "metrics unavailable"))
    else:
        error_count = int(metrics.get("prediction_error_count", 0))
        checks.append(_threshold_check("error_count", error_count, max_error_count))

    if drift_error is not None or drift is None:
        checks.append(_failed_check("drift_score", "drift unavailable"))
    else:
        drift_score = float(drift.get("drift_score", 0.0))
        checks.append(_threshold_check("drift_score", drift_score, max_drift_score))

    if metrics_error is not None or metrics is None:
        checks.append(_failed_check("latency_ms_last", "metrics unavailable"))
    else:
        latency_ms_last = float(metrics.get("prediction_latency_ms_last", 0.0))
        checks.append(_threshold_check("latency_ms_last", latency_ms_last, max_latency_ms_last))

    status = "healthy" if all(bool(check["passed"]) for check in checks) else "unhealthy"
    return {"status": status, "checks": checks}


def _parse_maximums(raw_maximums: list[str]) -> dict[str, float]:
    maximums = {}
    for raw_maximum in raw_maximums:
        metric_name, maximum = raw_maximum.split("=", maxsplit=1)
        maximums[metric_name] = float(maximum)
    return maximums


def main() -> None:
    parser = argparse.ArgumentParser(description="Check serving health, metrics, and drift thresholds.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-error-count", type=int, default=0)
    parser.add_argument("--max-drift-score", type=float, default=0.2)
    parser.add_argument("--max-latency-ms-last", type=float, default=100.0)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--maximum", action="append", default=[])
    args = parser.parse_args()

    if args.metrics_path is not None:
        if args.output_path is None:
            parser.error("--output-path is required with --metrics-path")
        summary = run_monitoring_loop(
            metrics_path=args.metrics_path,
            output_path=args.output_path,
            maximums=_parse_maximums(args.maximum),
        )
    else:
        summary = evaluate_monitoring_summary(
            base_url=args.base_url,
            max_error_count=args.max_error_count,
            max_drift_score=args.max_drift_score,
            max_latency_ms_last=args.max_latency_ms_last,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
