from __future__ import annotations

import json
from pathlib import Path


def read_processed_utterance_ids(paths: list[Path]) -> set[str]:
    processed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            utterance_id = row.get("source_utterance_id") or row.get("utterance_id")
            if utterance_id is None:
                continue
            processed.add(_source_utterance_id(utterance_id))
    return processed


def filter_pending_rows(rows: list[dict], processed_utterance_ids: set[str]) -> list[dict]:
    return [row for row in rows if row["utterance_id"] not in processed_utterance_ids]


def _source_utterance_id(utterance_id: str) -> str:
    if utterance_id.endswith("_p00"):
        return utterance_id
    if "_p" in utterance_id:
        return utterance_id.rsplit("_p", maxsplit=1)[0]
    return utterance_id
