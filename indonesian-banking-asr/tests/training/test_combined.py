from indonesian_banking_asr.training.combined import build_combined_training_rows, build_combined_training_summary


def test_build_combined_training_rows_keeps_70_20_10_mix():
    babelspeech_rows = [_row(f"real-{index}", source="babel", split="train") for index in range(10)]
    synthetic_rows = [
        *[_row(f"clean-{index}", source="synthetic", split="train", dataset_variant="clean") for index in range(10)],
        *[_row(f"aug-{index}", source="synthetic", split="train", dataset_variant="augmented") for index in range(10)],
        _row("heldout", source="synthetic", split="test", dataset_variant="clean"),
    ]

    rows = build_combined_training_rows(babelspeech_rows, synthetic_rows, seed=7)
    summary = build_combined_training_summary(rows)

    assert summary == {
        "total_rows": 14,
        "training_source_counts": {
            "babelspeech_general": 10,
            "synthetic_banking_clean": 3,
            "synthetic_banking_augmented": 1,
        },
        "dataset_variant_counts": {"real": 10, "clean": 3, "augmented": 1},
        "split_counts": {"train": 14},
    }
    assert {row["utterance_id"] for row in rows}.isdisjoint({"heldout"})


def test_build_combined_training_rows_limited_by_smallest_ratio_bucket():
    babelspeech_rows = [_row(f"real-{index}", source="babel", split="train") for index in range(100)]
    synthetic_rows = [
        *[_row(f"clean-{index}", source="synthetic", split="train", dataset_variant="clean") for index in range(4)],
        *[_row(f"aug-{index}", source="synthetic", split="train", dataset_variant="augmented") for index in range(1)],
    ]

    rows = build_combined_training_rows(babelspeech_rows, synthetic_rows, seed=7)

    assert build_combined_training_summary(rows)["training_source_counts"] == {
        "babelspeech_general": 7,
        "synthetic_banking_clean": 2,
        "synthetic_banking_augmented": 1,
    }


def _row(utterance_id, *, source, split, dataset_variant=None):
    row = {
        "utterance_id": utterance_id,
        "audio_path": f"audio/{utterance_id}.wav",
        "text": "halo dunia",
        "split": split,
        "source": source,
    }
    if dataset_variant:
        row["dataset_variant"] = dataset_variant
    return row
