"""End-to-end local RT transport + warehouse demo."""

from __future__ import annotations

import argparse
import json

from .rt_transport import RecommendationRequestEvent, consume_events, produce_events
from .warehouse import (
    insert_recommendation_result,
    processed_row_from_event,
    read_recommendation_results,
    reset_demo_rows,
)


def run_demo(event_count: int = 3, top_k: int = 5) -> dict[str, object]:
    reset_demo_rows()
    events = [RecommendationRequestEvent.sample(user_id=1000 + index, top_k=top_k) for index in range(event_count)]
    produced = produce_events(events)
    consumed_events = consume_events(
        expected_count=event_count,
        request_ids={event.request_id for event in events},
    )

    for event in consumed_events:
        insert_recommendation_result(
            processed_row_from_event(
                request_id=event.request_id,
                user_id=event.user_id,
                top_k=event.top_k,
                requested_at=event.requested_at,
            )
        )

    rows = read_recommendation_results()
    return {
        "produced": produced,
        "consumed": len(consumed_events),
        "warehouse_rows": len(rows),
        "rows": [row.__dict__ for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Redpanda + PostgreSQL RT foundation demo.")
    parser.add_argument("--event-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_demo(event_count=args.event_count, top_k=args.top_k), indent=2))


if __name__ == "__main__":
    main()
