from indonesian_banking_asr.evaluation import whisper


def test_transcribe_manifest_rows_writes_prediction_fields(monkeypatch):
    calls = []

    def fake_transcribe(audio_path, **kwargs):
        calls.append((audio_path, kwargs))
        return {"text": " halo dunia "}

    monkeypatch.setattr(whisper.mlx_whisper, "transcribe", fake_transcribe)

    rows = whisper.transcribe_manifest_rows(
        [{"utterance_id": "utt-1", "audio_path": "audio.wav"}],
        model="test-model",
        language="id",
    )

    assert rows == [
        {
            "utterance_id": "utt-1",
            "hypothesis": "halo dunia",
            "model": "test-model",
            "language": "id",
        }
    ]
    assert calls == [("audio.wav", {"path_or_hf_repo": "test-model", "language": "id", "verbose": False})]
