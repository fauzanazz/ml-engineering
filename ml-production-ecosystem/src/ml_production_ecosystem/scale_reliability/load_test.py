from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
from time import perf_counter, sleep
from collections.abc import Sequence
from typing import Callable
from urllib import error, request

DEFAULT_ENDPOINT = "/predict/v1"
DEFAULT_PAYLOAD = {"user_id": 10, "top_k": 5}

RequestResult = tuple[int, float]
Requester = Callable[[str, float], RequestResult]
AttemptResult = tuple[bool, float, int, int, str | None, bool]


def calculate_latency_summary(latency_ms_values: list[float]) -> dict[str, float]:
    if not latency_ms_values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    sorted_values = sorted(latency_ms_values)
    return {
        "min": round(sorted_values[0], 6),
        "p50": round(_nearest_rank(sorted_values, 50), 6),
        "p95": round(_nearest_rank(sorted_values, 95), 6),
        "max": round(sorted_values[-1], 6),
    }


def run_load_test(
    base_url: str,
    request_count: int,
    timeout_seconds: float,
    output_path: Path | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    requester: Requester | None = None,
    concurrency: int = 1,
    retry_count: int = 0,
    retry_delay_seconds: float = 0.0,
) -> dict[str, object]:
    if request_count < 1:
        raise ValueError("request_count must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if retry_count < 0:
        raise ValueError("retry_count must be at least 0")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be at least 0")

    request_url = f"{base_url.rstrip('/')}{endpoint}"
    send_request = requester or _post_prediction
    success_latencies: list[float] = []
    error_count = 0
    errors: dict[str, int] = {}

    if concurrency == 1:
        results = [
            _request_with_retries(send_request, request_url, timeout_seconds, retry_count, retry_delay_seconds)
            for _ in range(request_count)
        ]
    else:
        worker_count = min(concurrency, request_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_request_with_retries, send_request, request_url, timeout_seconds, retry_count, retry_delay_seconds)
                for _ in range(request_count)
            ]
            results = [future.result() for future in as_completed(futures)]

    for success, latency_ms, attempt_count, retry_attempt_count, error_key, retried in results:
        if success:
            success_latencies.append(latency_ms)
        else:
            error_count += 1
            if error_key is not None:
                errors[error_key] = errors.get(error_key, 0) + 1

    success_count = len(success_latencies)
    total_attempt_count = sum(result[2] for result in results)
    total_retry_attempt_count = sum(result[3] for result in results)
    retried_request_count = sum(1 for result in results if result[5])
    retry_success_count = sum(1 for result in results if result[0] and result[5])
    retry_exhausted_count = sum(1 for result in results if not result[0] and result[5])
    report = {
        "base_url": base_url.rstrip("/"),
        "endpoint": endpoint,
        "request_count": request_count,
        "concurrency": concurrency,
        "timeout_seconds": float(timeout_seconds),
        "retry_count": retry_count,
        "retry_delay_seconds": float(retry_delay_seconds),
        "success_count": success_count,
        "error_count": error_count,
        "attempt_count": total_attempt_count,
        "retry_attempt_count": total_retry_attempt_count,
        "retried_request_count": retried_request_count,
        "retry_success_count": retry_success_count,
        "retry_exhausted_count": retry_exhausted_count,
        "latency_ms": calculate_latency_summary(success_latencies),
        "errors": errors,
        "status": _status(success_count, error_count),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    return report


def parse_load_test_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small sequential or concurrent load test against serving API.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--request-count", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--retry-count", type=int, default=0)
    parser.add_argument("--retry-delay-seconds", type=float, default=0.0)
    parser.add_argument("--output-path", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_load_test_args()

    report = run_load_test(
        base_url=args.base_url,
        request_count=args.request_count,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        retry_count=args.retry_count,
        retry_delay_seconds=args.retry_delay_seconds,
        output_path=args.output_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _nearest_rank(sorted_values: list[float], percentile: int) -> float:
    rank = math.ceil((percentile / 100) * len(sorted_values))
    index = min(max(rank - 1, 0), len(sorted_values) - 1)
    return sorted_values[index]


def _request_with_retries(
    send_request: Requester,
    url: str,
    timeout_seconds: float,
    retry_count: int,
    retry_delay_seconds: float,
) -> AttemptResult:
    attempt_count = 0
    total_latency_ms = 0.0
    last_error_key: str | None = None

    while True:
        attempt_count += 1
        try:
            status_code, attempt_latency_ms = send_request(url, timeout_seconds)
            total_latency_ms += attempt_latency_ms
            if 200 <= status_code < 300:
                return True, total_latency_ms, attempt_count, attempt_count - 1, None, attempt_count > 1
            last_error_key = f"http_{status_code}"
            can_retry = status_code >= 500
        except TimeoutError:
            total_latency_ms += timeout_seconds * 1000
            last_error_key = "timeout"
            can_retry = True
        except (ConnectionError, OSError):
            last_error_key = "connection_error"
            can_retry = True
        except Exception:
            last_error_key = "request_error"
            can_retry = True

        if not can_retry or attempt_count > retry_count:
            return False, total_latency_ms, attempt_count, attempt_count - 1, last_error_key, attempt_count > 1
        if retry_delay_seconds > 0:
            total_latency_ms += retry_delay_seconds * 1000
            sleep(retry_delay_seconds)


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000


def _post_prediction(url: str, timeout_seconds: float) -> RequestResult:
    body = json.dumps(DEFAULT_PAYLOAD).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = perf_counter()
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response.read()
            status_code = response.status
    except error.HTTPError as http_error:
        http_error.read()
        status_code = http_error.code
    latency_ms = (perf_counter() - started_at) * 1000
    return status_code, latency_ms


def _status(success_count: int, error_count: int) -> str:
    if success_count == 0:
        return "failed"
    if error_count > 0:
        return "degraded"
    return "passed"
