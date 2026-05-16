from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from indonesian_banking_asr.synthetic.audit import write_jsonl

DEFAULT_BABELSPEECH_RATIO = 0.7
DEFAULT_SYNTHETIC_CLEAN_RATIO = 0.2
DEFAULT_SYNTHETIC_AUGMENTED_RATIO = 0.1


def build_combined_training_rows(
    babelspeech_rows: Iterable[dict],
    synthetic_rows: Iterable[dict],
    *,
    seed: int = 42,
    babelspeech_ratio: float = DEFAULT_BABELSPEECH_RATIO,
    synthetic_clean_ratio: float = DEFAULT_SYNTHETIC_CLEAN_RATIO,
    synthetic_augmented_ratio: float = DEFAULT_SYNTHETIC_AUGMENTED_RATIO,
) -> list[dict]:
    _validate_ratios(babelspeech_ratio, synthetic_clean_ratio, synthetic_augmented_ratio)
    train_babelspeech_rows = _training_rows(babelspeech_rows)
    train_clean_rows = [row for row in _training_rows(synthetic_rows) if row.get("dataset_variant") == "clean"]
    train_augmented_rows = [row for row in _training_rows(synthetic_rows) if row.get("dataset_variant") == "augmented"]

    sample_counts = _sample_counts(
        len(train_babelspeech_rows),
        len(train_clean_rows),
        len(train_augmented_rows),
        babelspeech_ratio=babelspeech_ratio,
        synthetic_clean_ratio=synthetic_clean_ratio,
        synthetic_augmented_ratio=synthetic_augmented_ratio,
    )
    rng = random.Random(seed)
    combined_rows = []
    combined_rows.extend(
        _tag_training_rows(
            _sample_rows(train_babelspeech_rows, sample_counts["babelspeech"], rng),
            training_source="babelspeech_general",
        )
    )
    combined_rows.extend(
        _tag_training_rows(
            _sample_rows(train_clean_rows, sample_counts["synthetic_clean"], rng),
            training_source="synthetic_banking_clean",
        )
    )
    combined_rows.extend(
        _tag_training_rows(
            _sample_rows(train_augmented_rows, sample_counts["synthetic_augmented"], rng),
            training_source="synthetic_banking_augmented",
        )
    )
    rng.shuffle(combined_rows)
    return combined_rows


def build_combined_training_summary(rows: Iterable[dict]) -> dict:
    row_list = list(rows)
    return {
        "total_rows": len(row_list),
        "training_source_counts": dict(Counter(row["training_source"] for row in row_list)),
        "dataset_variant_counts": dict(Counter(row.get("dataset_variant", "real") for row in row_list)),
        "split_counts": dict(Counter(row.get("split", "<missing>") for row in row_list)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined BabelSpeech + synthetic banking training manifest.")
    parser.add_argument("--babelspeech-manifest-path", required=True, type=Path)
    parser.add_argument("--synthetic-manifest-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_combined_training_rows(
        _read_jsonl(args.babelspeech_manifest_path),
        _read_jsonl(args.synthetic_manifest_path),
        seed=args.seed,
    )
    write_jsonl(args.output_path, rows)
    write_jsonl(args.summary_path, [build_combined_training_summary(rows)])


def _sample_counts(
    available_babelspeech: int,
    available_clean: int,
    available_augmented: int,
    *,
    babelspeech_ratio: float,
    synthetic_clean_ratio: float,
    synthetic_augmented_ratio: float,
) -> dict[str, int]:
    max_total = min(
        available_babelspeech / babelspeech_ratio,
        available_clean / synthetic_clean_ratio,
        available_augmented / synthetic_augmented_ratio,
    )
    total_rows = int(max_total)
    return {
        "babelspeech": min(available_babelspeech, round(total_rows * babelspeech_ratio)),
        "synthetic_clean": min(available_clean, round(total_rows * synthetic_clean_ratio)),
        "synthetic_augmented": min(available_augmented, round(total_rows * synthetic_augmented_ratio)),
    }


def _training_rows(rows: Iterable[dict]) -> list[dict]:
    return [row for row in rows if row.get("split") == "train"]


def _sample_rows(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if count > len(rows):
        raise ValueError("sample count exceeds available rows")
    return rng.sample(rows, count)


def _tag_training_rows(rows: list[dict], *, training_source: str) -> list[dict]:
    return [{**row, "training_source": training_source} for row in rows]


def _validate_ratios(*ratios: float) -> None:
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("training ratios must be positive")
    if round(sum(ratios), 10) != 1.0:
        raise ValueError("training ratios must sum to 1.0")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    main()
