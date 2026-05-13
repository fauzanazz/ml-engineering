from indonesian_banking_asr.synthetic.tts import SyntheticToneTts, build_audio_manifest_rows


def test_build_audio_manifest_rows_writes_wav_and_adds_audio_metadata(tmp_path):
    rows = [
        {
            "utterance_id": "utt-001",
            "text": "Saya mau cek saldo rekening 1234567890.",
        }
    ]
    audio_dir = tmp_path / "audio"

    audio_rows = build_audio_manifest_rows(
        rows,
        audio_dir=audio_dir,
        tts=SyntheticToneTts(sample_rate=8000, duration_sec=0.25),
    )

    audio_path = audio_dir / "utt-001.wav"
    assert audio_path.exists()
    assert audio_rows == [
        {
            "utterance_id": "utt-001",
            "text": "Saya mau cek saldo rekening 1234567890.",
            "audio_path": str(audio_path),
            "duration_sec": 0.25,
            "sample_rate": 8000,
            "tts_engine": "synthetic-tone",
        }
    ]


def test_build_audio_manifest_rows_rejects_missing_text(tmp_path):
    rows = [{"utterance_id": "utt-001"}]

    try:
        build_audio_manifest_rows(
            rows,
            audio_dir=tmp_path / "audio",
            tts=SyntheticToneTts(sample_rate=8000, duration_sec=0.25),
        )
    except ValueError as error:
        assert str(error) == "row utt-001 missing text"
    else:
        raise AssertionError("expected missing text to fail")
