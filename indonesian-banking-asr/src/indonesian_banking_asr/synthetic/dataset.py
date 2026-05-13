from __future__ import annotations

from collections import Counter
from typing import Iterable


def merge_audio_manifest_rows(clean_rows: Iterable[dict], augmented_rows: Iterable[dict]) -> list[dict]:
    merged_rows = []
    seen_utterance_ids = set()

    for row in clean_rows:
        merged_rows.append(_tag_row(row, "clean", seen_utterance_ids))
    for row in augmented_rows:
        merged_rows.append(_tag_row(row, "augmented", seen_utterance_ids))

    return merged_rows


def build_dataset_summary(rows: Iterable[dict]) -> dict:
    row_list = list(rows)
    return {
        "total_rows": len(row_list),
        "dataset_variant_counts": dict(Counter(row["dataset_variant"] for row in row_list)),
        "split_counts": dict(Counter(row["split"] for row in row_list if "split" in row)),
        "augmentation_profile_counts": dict(
            Counter(
                row["augmentation"]["name"]
                for row in row_list
                if row.get("augmentation", {}).get("name")
            )
        ),
    }


def validate_dataset_manifest_rows(rows: Iterable[dict]) -> dict:
    errors = []
    checked_rows = 0
    invalid_utterance_ids = set()

    for row in rows:
        checked_rows += 1
        utterance_id = row.get("utterance_id", "<missing>")
        for field in ("utterance_id", "text", "audio_path", "dataset_variant"):
            if not row.get(field):
                errors.append(
                    {
                        "utterance_id": utterance_id,
                        "field": field,
                        "message": "required field missing",
                    }
                )
                invalid_utterance_ids.add(utterance_id)

        if row.get("dataset_variant") == "augmented":
            for field in ("source_audio_path", "augmentation"):
                if not row.get(field):
                    errors.append(
                        {
                            "utterance_id": utterance_id,
                            "field": field,
                            "message": "required for augmented rows",
                        }
                    )
                    invalid_utterance_ids.add(utterance_id)

    invalid_rows = len(invalid_utterance_ids)
    return {
        "checked_rows": checked_rows,
        "valid_rows": checked_rows - invalid_rows,
        "invalid_rows": invalid_rows,
        "errors": errors,
    }



def _tag_row(row: dict, dataset_variant: str, seen_utterance_ids: set[str]) -> dict:
    utterance_id = row["utterance_id"]
    if utterance_id in seen_utterance_ids:
        raise ValueError(f"duplicate utterance_id: {utterance_id}")
    seen_utterance_ids.add(utterance_id)
    return {**row, "dataset_variant": dataset_variant}
