from __future__ import annotations

from collections import Counter


def build_generation_summary(
    canonical_rows: list[dict],
    pending_rows: list[dict],
    accepted_rows: list[dict],
    rejected_rows: list[dict],
    raw_rows: list[dict],
    skipped_count: int,
) -> dict:
    return {
        "canonical_rows": len(canonical_rows),
        "pending_rows": len(pending_rows),
        "skipped_rows": skipped_count,
        "accepted_rows": len(accepted_rows),
        "rejected_rows": len(rejected_rows),
        "raw_rows": len(raw_rows),
        "raw_status_counts": dict(Counter(row.get("status") for row in raw_rows)),
        "split_counts": dict(Counter(row.get("split") for row in canonical_rows)),
    }
