from indonesian_banking_asr.synthetic.dataset import build_dataset_summary, merge_audio_manifest_rows


def test_merge_audio_manifest_rows_combines_clean_and_augmented_rows():
    clean_rows = [
        {
            "utterance_id": "utt-001",
            "audio_path": "audio/utt-001.wav",
            "text": "Saya mau cek saldo.",
        }
    ]
    augmented_rows = [
        {
            "utterance_id": "utt-001_aug_noisy",
            "audio_path": "augmented/utt-001_aug_noisy.wav",
            "source_audio_path": "audio/utt-001.wav",
            "text": "Saya mau cek saldo.",
            "augmentation": {"name": "noisy", "gain": 1.0, "noise_amplitude": 200, "seed": 7},
        }
    ]

    rows = merge_audio_manifest_rows(clean_rows, augmented_rows)

    assert rows == [
        {
            "utterance_id": "utt-001",
            "audio_path": "audio/utt-001.wav",
            "text": "Saya mau cek saldo.",
            "dataset_variant": "clean",
        },
        {
            "utterance_id": "utt-001_aug_noisy",
            "audio_path": "augmented/utt-001_aug_noisy.wav",
            "source_audio_path": "audio/utt-001.wav",
            "text": "Saya mau cek saldo.",
            "augmentation": {"name": "noisy", "gain": 1.0, "noise_amplitude": 200, "seed": 7},
            "dataset_variant": "augmented",
        },
    ]


def test_merge_audio_manifest_rows_rejects_duplicate_utterance_ids():
    clean_rows = [{"utterance_id": "utt-001", "audio_path": "audio/utt-001.wav"}]
    augmented_rows = [{"utterance_id": "utt-001", "audio_path": "augmented/utt-001.wav"}]

    try:
        merge_audio_manifest_rows(clean_rows, augmented_rows)
    except ValueError as error:
        assert str(error) == "duplicate utterance_id: utt-001"
    else:
        raise AssertionError("expected duplicate utterance_id to fail")


def test_build_dataset_summary_counts_variants_splits_and_augmentation_profiles():
    rows = [
        {
            "utterance_id": "utt-001",
            "dataset_variant": "clean",
            "split": "train",
        },
        {
            "utterance_id": "utt-002",
            "dataset_variant": "clean",
            "split": "test",
        },
        {
            "utterance_id": "utt-001_aug_quiet",
            "dataset_variant": "augmented",
            "split": "train",
            "augmentation": {"name": "quiet", "gain": 0.5, "noise_amplitude": 0, "seed": 7},
        },
        {
            "utterance_id": "utt-001_aug_noisy",
            "dataset_variant": "augmented",
            "split": "train",
            "augmentation": {"name": "noisy", "gain": 1.0, "noise_amplitude": 200, "seed": 8},
        },
    ]

    summary = build_dataset_summary(rows)

    assert summary == {
        "total_rows": 4,
        "dataset_variant_counts": {"clean": 2, "augmented": 2},
        "split_counts": {"train": 3, "test": 1},
        "augmentation_profile_counts": {"quiet": 1, "noisy": 1},
    }
