import wave

from indonesian_banking_asr.synthetic.audio_qa import validate_audio_manifest_rows
from indonesian_banking_asr.synthetic.tts import SyntheticToneTts, build_audio_manifest_rows


def test_validate_audio_manifest_rows_accepts_valid_wav(tmp_path):
    rows = build_audio_manifest_rows(
        [
            {
                "utterance_id": "utt-001",
                "text": "Saya mau cek saldo rekening 1234567890.",
            }
        ],
        audio_dir=tmp_path / "audio",
        tts=SyntheticToneTts(sample_rate=8000, duration_sec=0.25),
    )

    report = validate_audio_manifest_rows(rows)

    assert report == {
        "checked_rows": 1,
        "valid_rows": 1,
        "invalid_rows": 0,
        "errors": [],
    }


def test_validate_audio_manifest_rows_reports_missing_audio_file(tmp_path):
    rows = [
        {
            "utterance_id": "utt-001",
            "audio_path": str(tmp_path / "missing.wav"),
            "duration_sec": 0.25,
            "sample_rate": 8000,
        }
    ]

    report = validate_audio_manifest_rows(rows)

    assert report == {
        "checked_rows": 1,
        "valid_rows": 0,
        "invalid_rows": 1,
        "errors": [
            {
                "utterance_id": "utt-001",
                "field": "audio_path",
                "message": "audio file missing",
            }
        ],
    }


def test_validate_audio_manifest_rows_reports_sample_rate_mismatch(tmp_path):
    rows = build_audio_manifest_rows(
        [
            {
                "utterance_id": "utt-001",
                "text": "Saya mau cek saldo rekening 1234567890.",
            }
        ],
        audio_dir=tmp_path / "audio",
        tts=SyntheticToneTts(sample_rate=8000, duration_sec=0.25),
    )
    rows[0]["sample_rate"] = 16000

    report = validate_audio_manifest_rows(rows)

    assert report["checked_rows"] == 1
    assert report["valid_rows"] == 0
    assert report["invalid_rows"] == 1
    assert report["errors"] == [
        {
            "utterance_id": "utt-001",
            "field": "sample_rate",
            "message": "expected 16000, got 8000",
        }
    ]


def test_validate_audio_manifest_rows_reports_duration_mismatch(tmp_path):
    rows = build_audio_manifest_rows(
        [
            {
                "utterance_id": "utt-001",
                "text": "Saya mau cek saldo rekening 1234567890.",
            }
        ],
        audio_dir=tmp_path / "audio",
        tts=SyntheticToneTts(sample_rate=8000, duration_sec=0.25),
    )
    rows[0]["duration_sec"] = 1.0

    report = validate_audio_manifest_rows(rows)

    assert report["checked_rows"] == 1
    assert report["valid_rows"] == 0
    assert report["invalid_rows"] == 1
    assert report["errors"] == [
        {
            "utterance_id": "utt-001",
            "field": "duration_sec",
            "message": "expected 1.0, got 0.25",
        }
    ]


def test_validate_audio_manifest_rows_reports_corrupted_wav(tmp_path):
    audio_path = tmp_path / "broken.wav"
    audio_path.write_text("not a wav")
    rows = [
        {
            "utterance_id": "utt-001",
            "audio_path": str(audio_path),
            "duration_sec": 0.25,
            "sample_rate": 8000,
        }
    ]

    report = validate_audio_manifest_rows(rows)

    assert report == {
        "checked_rows": 1,
        "valid_rows": 0,
        "invalid_rows": 1,
        "errors": [
            {
                "utterance_id": "utt-001",
                "field": "audio_path",
                "message": "invalid wav file",
            }
        ],
    }


def test_validate_audio_manifest_rows_reports_silent_wav(tmp_path):
    audio_path = tmp_path / "silent.wav"
    sample_rate = 8000
    duration_sec = 0.25
    frame_count = int(sample_rate * duration_sec)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    rows = [
        {
            "utterance_id": "utt-001",
            "audio_path": str(audio_path),
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
        }
    ]

    report = validate_audio_manifest_rows(rows)

    assert report == {
        "checked_rows": 1,
        "valid_rows": 0,
        "invalid_rows": 1,
        "errors": [
            {
                "utterance_id": "utt-001",
                "field": "audio_path",
                "message": "audio is silent",
            }
        ],
    }
