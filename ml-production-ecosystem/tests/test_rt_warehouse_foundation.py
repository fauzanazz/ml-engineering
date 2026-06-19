import os
import socket

import pytest

from ml_production_ecosystem.recommendation.rt_demo import run_demo
from ml_production_ecosystem.recommendation.rt_transport import (
    RecommendationRequestEvent,
    consume_events,
    produce_events,
)
from ml_production_ecosystem.recommendation.warehouse import (
    RecommendationRequestRow,
    insert_recommendation_result,
    read_recommendation_results,
    reset_demo_rows,
)


def _service_reachable(host: str, port: int, timeout_seconds: float = 0.5) -> bool:
    with socket.socket() as sock:
        sock.settimeout(timeout_seconds)
        try:
            sock.connect((host, port))
        except OSError:
            return False
    return True


def _integration_enabled() -> bool:
    return os.environ.get("ML_ECOSYSTEM_RUN_RT_WAREHOUSE_TESTS", "").lower() in {"1", "true", "yes", "on"}


def _require_services() -> None:
    if not _integration_enabled():
        pytest.skip(
            "RT warehouse integration tests require services; set ML_ECOSYSTEM_RUN_RT_WAREHOUSE_TESTS=1 and start docker-compose redpanda+postgres"
        )

    if not _service_reachable("127.0.0.1", 9092):
        pytest.skip("Redpanda is not reachable on 127.0.0.1:9092")

    if not _service_reachable("127.0.0.1", 5432):
        pytest.skip("PostgreSQL is not reachable on 127.0.0.1:5432")


def test_redpanda_transport_produces_and_consumes_recommendation_request() -> None:
    _require_services()

    event = RecommendationRequestEvent.sample(user_id=42, top_k=3)

    produced = produce_events([event])
    consumed = consume_events(expected_count=1, request_ids={event.request_id})

    assert produced == 1
    assert consumed[-1].request_id == event.request_id
    assert consumed[-1].user_id == 42
    assert consumed[-1].top_k == 3


def test_postgres_warehouse_creates_inserts_and_reads_recommendation_result() -> None:
    _require_services()

    reset_demo_rows()
    row = RecommendationRequestRow(
        request_id="warehouse-test-request",
        user_id=7,
        top_k=5,
        requested_at="2026-05-12T00:00:00+00:00",
        processed_at="2026-05-12T00:00:01+00:00",
        recommendation_count=5,
    )

    insert_recommendation_result(row)
    rows = read_recommendation_results()

    assert len(rows) == 1
    assert rows[0].request_id == "warehouse-test-request"
    assert rows[0].recommendation_count == 5


def test_rt_demo_writes_consumed_events_to_warehouse() -> None:
    _require_services()

    result = run_demo(event_count=2, top_k=4)

    assert result["produced"] == 2
    assert result["consumed"] == 2
    assert result["warehouse_rows"] == 2
    assert {row["recommendation_count"] for row in result["rows"]} == {4}
