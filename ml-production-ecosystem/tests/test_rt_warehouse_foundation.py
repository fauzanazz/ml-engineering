from recommendation.rt_demo import run_demo
from recommendation.rt_transport import (
    RecommendationRequestEvent,
    consume_events,
    produce_events,
)
from recommendation.warehouse import (
    RecommendationRequestRow,
    insert_recommendation_result,
    read_recommendation_results,
    reset_demo_rows,
)


def test_redpanda_transport_produces_and_consumes_recommendation_request() -> None:
    event = RecommendationRequestEvent.sample(user_id=42, top_k=3)

    produced = produce_events([event])
    consumed = consume_events(expected_count=1, request_ids={event.request_id})

    assert produced == 1
    assert consumed[-1].request_id == event.request_id
    assert consumed[-1].user_id == 42
    assert consumed[-1].top_k == 3


def test_postgres_warehouse_creates_inserts_and_reads_recommendation_result() -> None:
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
    result = run_demo(event_count=2, top_k=4)

    assert result["produced"] == 2
    assert result["consumed"] == 2
    assert result["warehouse_rows"] == 2
    assert {row["recommendation_count"] for row in result["rows"]} == {4}
