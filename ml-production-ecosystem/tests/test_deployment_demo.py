from pathlib import Path
import json
from typing import Any

from ml_production_ecosystem.production_patterns.deployment_demo import build_deployment_demo_report


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = 200
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def get(self, url: str, timeout: float) -> FakeResponse:
        if url.endswith("/health"):
            return FakeResponse({"status": "ok"})
        if url.endswith("/metrics.json"):
            return FakeResponse({"prediction_error_count": 0, "prediction_latency_ms_last": 12.0})
        if url.endswith("/drift"):
            return FakeResponse({"drift_score": 0.1})
        raise AssertionError(f"unexpected URL: {url}")


def test_deployment_demo_skips_without_base_url(tmp_path: Path) -> None:
    output_path = tmp_path / "deployment-demo.json"

    report = build_deployment_demo_report(None, output_path)

    assert report == {"status": "skipped", "reason": "base URL not provided", "checks": []}
    assert json.loads(output_path.read_text()) == report


def test_deployment_demo_passes_healthy_local_api(tmp_path: Path) -> None:
    output_path = tmp_path / "deployment-demo.json"

    report = build_deployment_demo_report(
        "http://127.0.0.1:8000",
        output_path,
        http_client=FakeClient(),
    )

    assert report["status"] == "passed"
    assert report["base_url"] == "http://127.0.0.1:8000"
    assert {check["name"] for check in report["checks"]} == {
        "health",
        "error_count",
        "drift_score",
        "latency_ms_last",
    }
    assert json.loads(output_path.read_text()) == report
