"""Kafka-compatible real-time transport foundation for recommendation requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from typing import Iterable
from uuid import uuid4

from confluent_kafka import Consumer, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

RECOMMENDATION_REQUEST_TOPIC = "foundation.recommendation.requests"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"


@dataclass(frozen=True)
class RecommendationRequestEvent:
    request_id: str
    user_id: int
    top_k: int
    requested_at: str

    @classmethod
    def sample(cls, user_id: int, top_k: int = 5) -> "RecommendationRequestEvent":
        return cls(
            request_id=str(uuid4()),
            user_id=user_id,
            top_k=top_k,
            requested_at=datetime.now(UTC).isoformat(),
        )


def serialize_event(event: RecommendationRequestEvent) -> bytes:
    return json.dumps(asdict(event), sort_keys=True).encode("utf-8")


def deserialize_event(payload: bytes) -> RecommendationRequestEvent:
    data = json.loads(payload.decode("utf-8"))
    return RecommendationRequestEvent(
        request_id=str(data["request_id"]),
        user_id=int(data["user_id"]),
        top_k=int(data["top_k"]),
        requested_at=str(data["requested_at"]),
    )


def ensure_topic(
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    topic: str = RECOMMENDATION_REQUEST_TOPIC,
) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    futures = admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])
    future = futures[topic]
    try:
        future.result(timeout=15)
    except KafkaException as exc:
        if "TOPIC_ALREADY_EXISTS" not in str(exc):
            raise


def produce_events(
    events: Iterable[RecommendationRequestEvent],
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    topic: str = RECOMMENDATION_REQUEST_TOPIC,
) -> int:
    ensure_topic(bootstrap_servers, topic)
    producer = Producer({"bootstrap.servers": bootstrap_servers})
    count = 0
    for event in events:
        producer.produce(topic, key=event.request_id.encode("utf-8"), value=serialize_event(event))
        count += 1
    producer.flush(15)
    return count


def consume_events(
    expected_count: int,
    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
    topic: str = RECOMMENDATION_REQUEST_TOPIC,
    group_id: str | None = None,
    timeout_seconds: float = 20.0,
    request_ids: set[str] | None = None,
) -> list[RecommendationRequestEvent]:
    ensure_topic(bootstrap_servers, topic)
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id or f"foundation-rt-demo-{uuid4()}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    events: list[RecommendationRequestEvent] = []
    deadline = datetime.now(UTC).timestamp() + timeout_seconds
    try:
        while len(events) < expected_count and datetime.now(UTC).timestamp() < deadline:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                raise KafkaException(message.error())
            event = deserialize_event(message.value())
            if request_ids is not None and event.request_id not in request_ids:
                continue
            events.append(event)
    finally:
        consumer.close()
    return events
