"""PostgreSQL warehouse foundation for processed recommendation requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import os

import psycopg

DEFAULT_WAREHOUSE_DSN = "postgresql://mlops:mlops@localhost:5432/mlops_warehouse"


@dataclass(frozen=True)
class RecommendationRequestRow:
    request_id: str
    user_id: int
    top_k: int
    requested_at: str
    processed_at: str
    recommendation_count: int


def warehouse_dsn() -> str:
    return os.environ.get("WAREHOUSE_DSN", DEFAULT_WAREHOUSE_DSN)


def ensure_schema(dsn: str | None = None) -> None:
    with psycopg.connect(dsn or warehouse_dsn()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_request_results (
                request_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                top_k INTEGER NOT NULL,
                requested_at TIMESTAMPTZ NOT NULL,
                processed_at TIMESTAMPTZ NOT NULL,
                recommendation_count INTEGER NOT NULL
            )
            """
        )


def reset_demo_rows(dsn: str | None = None) -> None:
    ensure_schema(dsn)
    with psycopg.connect(dsn or warehouse_dsn()) as connection:
        connection.execute("DELETE FROM recommendation_request_results")


def insert_recommendation_result(row: RecommendationRequestRow, dsn: str | None = None) -> None:
    ensure_schema(dsn)
    with psycopg.connect(dsn or warehouse_dsn()) as connection:
        connection.execute(
            """
            INSERT INTO recommendation_request_results (
                request_id, user_id, top_k, requested_at, processed_at, recommendation_count
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (request_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                top_k = EXCLUDED.top_k,
                requested_at = EXCLUDED.requested_at,
                processed_at = EXCLUDED.processed_at,
                recommendation_count = EXCLUDED.recommendation_count
            """,
            (
                row.request_id,
                row.user_id,
                row.top_k,
                row.requested_at,
                row.processed_at,
                row.recommendation_count,
            ),
        )


def read_recommendation_results(dsn: str | None = None) -> list[RecommendationRequestRow]:
    ensure_schema(dsn)
    with psycopg.connect(dsn or warehouse_dsn()) as connection:
        rows = connection.execute(
            """
            SELECT request_id, user_id, top_k, requested_at, processed_at, recommendation_count
            FROM recommendation_request_results
            ORDER BY processed_at, request_id
            """
        ).fetchall()
    return [
        RecommendationRequestRow(
            request_id=row[0],
            user_id=row[1],
            top_k=row[2],
            requested_at=row[3].isoformat(),
            processed_at=row[4].isoformat(),
            recommendation_count=row[5],
        )
        for row in rows
    ]


def processed_row_from_event(request_id: str, user_id: int, top_k: int, requested_at: str) -> RecommendationRequestRow:
    return RecommendationRequestRow(
        request_id=request_id,
        user_id=user_id,
        top_k=top_k,
        requested_at=requested_at,
        processed_at=datetime.now(UTC).isoformat(),
        recommendation_count=top_k,
    )
