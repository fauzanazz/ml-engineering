from ml_production_ecosystem.production_patterns.monitoring_loop import evaluate_monitoring_summary


class StubResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class StubHttpClient:
    def __init__(self, responses: dict[str, StubResponse]) -> None:
        self.responses = responses

    def get(self, url: str, timeout: float) -> StubResponse:
        return self.responses[url]


def test_monitoring_summary_is_healthy_when_endpoints_and_thresholds_pass() -> None:
    client = StubHttpClient(
        {
            "http://api.test/health": StubResponse(200, {"status": "ok"}),
            "http://api.test/metrics.json": StubResponse(
                200,
                {
                    "prediction_error_count": 0,
                    "prediction_latency_ms_last": 3.2,
                },
            ),
            "http://api.test/drift": StubResponse(200, {"drift_score": 0.0}),
        }
    )

    summary = evaluate_monitoring_summary(
        base_url="http://api.test",
        max_error_count=0,
        max_drift_score=0.2,
        max_latency_ms_last=100.0,
        http_client=client,
    )

    assert summary == {
        "status": "healthy",
        "checks": [
            {"name": "health", "passed": True, "message": "ok"},
            {"name": "error_count", "passed": True, "message": "0 <= 0"},
            {"name": "drift_score", "passed": True, "message": "0.0 <= 0.2"},
            {"name": "latency_ms_last", "passed": True, "message": "3.2 <= 100.0"},
        ],
    }


def test_monitoring_summary_is_unhealthy_when_threshold_fails() -> None:
    client = StubHttpClient(
        {
            "http://api.test/health": StubResponse(200, {"status": "ok"}),
            "http://api.test/metrics.json": StubResponse(
                200,
                {
                    "prediction_error_count": 2,
                    "prediction_latency_ms_last": 150.0,
                },
            ),
            "http://api.test/drift": StubResponse(200, {"drift_score": 0.3}),
        }
    )

    summary = evaluate_monitoring_summary(
        base_url="http://api.test",
        max_error_count=0,
        max_drift_score=0.2,
        max_latency_ms_last=100.0,
        http_client=client,
    )

    assert summary["status"] == "unhealthy"
    assert summary["checks"] == [
        {"name": "health", "passed": True, "message": "ok"},
        {"name": "error_count", "passed": False, "message": "2 <= 0"},
        {"name": "drift_score", "passed": False, "message": "0.3 <= 0.2"},
        {"name": "latency_ms_last", "passed": False, "message": "150.0 <= 100.0"},
    ]


def test_monitoring_summary_reports_http_failure_as_failed_check() -> None:
    class FailingHttpClient:
        def get(self, url: str, timeout: float) -> StubResponse:
            raise RuntimeError("connection refused")

    summary = evaluate_monitoring_summary(
        base_url="http://api.test",
        max_error_count=0,
        max_drift_score=0.2,
        max_latency_ms_last=100.0,
        http_client=FailingHttpClient(),
    )

    assert summary == {
        "status": "unhealthy",
        "checks": [
            {"name": "health", "passed": False, "message": "connection refused"},
            {"name": "error_count", "passed": False, "message": "metrics unavailable"},
            {"name": "drift_score", "passed": False, "message": "drift unavailable"},
            {"name": "latency_ms_last", "passed": False, "message": "metrics unavailable"},
        ],
    }
