from indonesian_banking_asr.synthetic.dataset import merge_audio_manifest_rows


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
