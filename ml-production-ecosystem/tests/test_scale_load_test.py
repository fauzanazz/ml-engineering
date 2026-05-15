from pathlib import Path
from threading import Lock
from time import sleep

import pytest

from scale_reliability.load_test import calculate_latency_summary, run_load_test


def test_run_load_test_reports_successes_and_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "load-test.json"

    def requester(url: str, timeout_seconds: float) -> tuple[int, float]:
        assert url == "http://127.0.0.1:8000/predict/v1"
        assert timeout_seconds == 2
        return 200, 8.0

    report = run_load_test(
        base_url="http://127.0.0.1:8000",
        request_count=3,
        timeout_seconds=2,
        output_path=output_path,
        requester=requester,
    )

    assert report == {
        "base_url": "http://127.0.0.1:8000",
        "endpoint": "/predict/v1",
        "request_count": 3,
        "concurrency": 1,
        "success_count": 3,
        "error_count": 0,
        "latency_ms": {"min": 8.0, "p50": 8.0, "p95": 8.0, "max": 8.0},
        "status": "passed",
    }
    assert output_path.read_text().strip().startswith("{")


def test_run_load_test_reports_partial_failure_without_crashing() -> None:
    outcomes = iter([(200, 5.0), (500, 7.0), TimeoutError("slow")])

    def requester(url: str, timeout_seconds: float) -> tuple[int, float]:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    report = run_load_test(
        base_url="http://127.0.0.1:8000/",
        request_count=3,
        timeout_seconds=1,
        requester=requester,
    )

    assert report["success_count"] == 1
    assert report["error_count"] == 2
    assert report["latency_ms"] == {"min": 5.0, "p50": 5.0, "p95": 5.0, "max": 5.0}
    assert report["status"] == "degraded"


def test_run_load_test_reports_full_failure_without_latency_values() -> None:
    def requester(url: str, timeout_seconds: float) -> tuple[int, float]:
        raise ConnectionError("offline")

    report = run_load_test(
        base_url="http://127.0.0.1:8000",
        request_count=2,
        timeout_seconds=1,
        requester=requester,
    )

    assert report["success_count"] == 0
    assert report["error_count"] == 2
    assert report["latency_ms"] == {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    assert report["status"] == "failed"


def test_run_load_test_runs_requests_concurrently_when_concurrency_is_greater_than_one() -> None:
    active_count = 0
    max_active_count = 0
    lock = Lock()

    def requester(url: str, timeout_seconds: float) -> tuple[int, float]:
        nonlocal active_count, max_active_count
        with lock:
            active_count += 1
            max_active_count = max(max_active_count, active_count)
        sleep(0.01)
        with lock:
            active_count -= 1
        return 200, 12.0

    report = run_load_test(
        base_url="http://127.0.0.1:8000",
        request_count=6,
        concurrency=3,
        timeout_seconds=1,
        requester=requester,
    )

    assert max_active_count > 1
    assert max_active_count <= 3
    assert report["concurrency"] == 3
    assert report["success_count"] == 6
    assert report["error_count"] == 0
    assert report["status"] == "passed"


def test_run_load_test_rejects_invalid_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        run_load_test(
            base_url="http://127.0.0.1:8000",
            request_count=1,
            concurrency=0,
            timeout_seconds=1,
        )


def test_calculate_latency_summary_uses_nearest_rank_percentiles() -> None:
    latencies = [30.0, 10.0, 20.0, 40.0, 50.0]

    assert calculate_latency_summary(latencies) == {
        "min": 10.0,
        "p50": 30.0,
        "p95": 50.0,
        "max": 50.0,
    }
