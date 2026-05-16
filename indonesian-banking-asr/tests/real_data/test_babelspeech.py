import json
from pathlib import Path

from indonesian_banking_asr.real_data.babelspeech import build_babelspeech_manifest_rows, read_metadata


def test_build_babelspeech_manifest_rows_maps_project_fields(tmp_path):
    dataset_root = tmp_path / "babelspeech"
    metadata_rows = [
        {
            "filename": f"sample_{row_index}.wav",
            "relative_path": f"wav/sample_{row_index}.wav",
            "duration": 1.5,
            "confidence": 0.99,
            "text": " halo bank indonesia ",
            "snr": 55.0,
            "dnsmos": 4.1,
        }
        for row_index in range(10)
    ]

    rows = build_babelspeech_manifest_rows(metadata_rows, dataset_root=dataset_root)

    assert rows[0] == {
        "utterance_id": "sample_0",
        "audio_path": str(dataset_root / "wav/sample_0.wav"),
        "text": "halo bank indonesia",
        "split": "train",
        "source": "BabelSpeech/40hours_Indonesian_Colloquial_ASR_Speech_Dataset",
        "duration": 1.5,
        "confidence": 0.99,
        "snr": 55.0,
        "dnsmos": 4.1,
    }
    assert [row["split"] for row in rows].count("train") == 8
    assert [row["split"] for row in rows].count("validation") == 1
    assert [row["split"] for row in rows].count("test") == 1


def test_build_babelspeech_manifest_rows_can_target_extracted_audio(tmp_path):
    dataset_root = tmp_path / "babelspeech"
    extracted_audio_root = tmp_path / "data" / "wav"
    metadata_rows = [{"relative_path": "wav/audio.wav", "text": "halo"}]

    rows = build_babelspeech_manifest_rows(
        metadata_rows,
        dataset_root=dataset_root,
        extracted_audio_root=extracted_audio_root,
    )

    assert rows[0]["audio_path"] == str(extracted_audio_root / "audio.wav")


def test_read_metadata_rejects_non_list(tmp_path):
    metadata_path = tmp_path / "audio_info.json"
    metadata_path.write_text(json.dumps({"rows": []}))

    try:
        read_metadata(metadata_path)
    except ValueError as error:
        assert str(error) == "BabelSpeech metadata must be JSON list"
    else:
        raise AssertionError("expected ValueError")
