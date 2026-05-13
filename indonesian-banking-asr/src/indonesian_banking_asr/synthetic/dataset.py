from __future__ import annotations

from typing import Iterable


def merge_audio_manifest_rows(clean_rows: Iterable[dict], augmented_rows: Iterable[dict]) -> list[dict]:
    merged_rows = []
    seen_utterance_ids = set()

    for row in clean_rows:
        merged_rows.append(_tag_row(row, "clean", seen_utterance_ids))
    for row in augmented_rows:
        merged_rows.append(_tag_row(row, "augmented", seen_utterance_ids))

    return merged_rows


def _tag_row(row: dict, dataset_variant: str, seen_utterance_ids: set[str]) -> dict:
    utterance_id = row["utterance_id"]
    if utterance_id in seen_utterance_ids:
        raise ValueError(f"duplicate utterance_id: {utterance_id}")
    seen_utterance_ids.add(utterance_id)
    return {**row, "dataset_variant": dataset_variant}
