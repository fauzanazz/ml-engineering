import wave

from indonesian_banking_asr.synthetic.augmentation import build_augmented_manifest_rows
from indonesian_banking_asr.synthetic.tts import SyntheticToneTts, build_audio_manifest_rows


def test_build_augmented_manifest_rows_writes_gain_augmented_wav(tmp_path):
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

    augmented_rows = build_augmented_manifest_rows(
        rows,
        output_dir=tmp_path / "augmented",
        gain=0.5,
    )

    augmented_path = tmp_path / "augmented" / "utt-001_gain-0.5.wav"
    assert augmented_path.exists()
    assert augmented_rows == [
        {
            **rows[0],
            "audio_path": str(augmented_path),
            "source_audio_path": rows[0]["audio_path"],
            "augmentation": {"gain": 0.5},
        }
    ]
    with wave.open(rows[0]["audio_path"], "rb") as original_wav:
        original_frames = original_wav.readframes(original_wav.getnframes())
    with wave.open(str(augmented_path), "rb") as augmented_wav:
        assert augmented_wav.getframerate() == 8000
        assert augmented_wav.getnchannels() == 1
        augmented_frames = augmented_wav.readframes(augmented_wav.getnframes())
    assert augmented_frames != original_frames


def test_build_augmented_manifest_rows_rejects_non_positive_gain(tmp_path):
    rows = [{"utterance_id": "utt-001", "audio_path": str(tmp_path / "audio.wav")}]

    try:
        build_augmented_manifest_rows(rows, output_dir=tmp_path / "augmented", gain=0)
    except ValueError as error:
        assert str(error) == "gain must be positive"
    else:
        raise AssertionError("expected invalid gain to fail")


def test_build_augmented_manifest_rows_adds_deterministic_noise(tmp_path):
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

    first_rows = build_augmented_manifest_rows(
        rows,
        output_dir=tmp_path / "augmented-1",
        gain=1.0,
        noise_amplitude=200,
        seed=7,
    )
    second_rows = build_augmented_manifest_rows(
        rows,
        output_dir=tmp_path / "augmented-2",
        gain=1.0,
        noise_amplitude=200,
        seed=7,
    )

    assert first_rows[0]["augmentation"] == {
        "gain": 1.0,
        "noise_amplitude": 200,
        "seed": 7,
    }
    with wave.open(first_rows[0]["audio_path"], "rb") as first_wav:
        first_frames = first_wav.readframes(first_wav.getnframes())
    with wave.open(second_rows[0]["audio_path"], "rb") as second_wav:
        second_frames = second_wav.readframes(second_wav.getnframes())
    with wave.open(rows[0]["audio_path"], "rb") as original_wav:
        original_frames = original_wav.readframes(original_wav.getnframes())
    assert first_frames == second_frames
    assert first_frames != original_frames


def test_build_augmented_manifest_rows_rejects_negative_noise(tmp_path):
    rows = [{"utterance_id": "utt-001", "audio_path": str(tmp_path / "audio.wav")}]

    try:
        build_augmented_manifest_rows(
            rows,
            output_dir=tmp_path / "augmented",
            gain=1.0,
            noise_amplitude=-1,
        )
    except ValueError as error:
        assert str(error) == "noise_amplitude cannot be negative"
    else:
        raise AssertionError("expected invalid noise amplitude to fail")


def test_build_augmented_manifest_rows_expands_multiple_profiles(tmp_path):
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

    augmented_rows = build_augmented_manifest_rows(
        rows,
        output_dir=tmp_path / "augmented",
        profiles=[
            {"name": "quiet", "gain": 0.5, "noise_amplitude": 0, "seed": 7},
            {"name": "noisy", "gain": 1.0, "noise_amplitude": 200, "seed": 8},
        ],
    )

    assert [row["utterance_id"] for row in augmented_rows] == [
        "utt-001_aug_quiet",
        "utt-001_aug_noisy",
    ]
    assert [row["augmentation"] for row in augmented_rows] == [
        {"name": "quiet", "gain": 0.5, "noise_amplitude": 0, "seed": 7},
        {"name": "noisy", "gain": 1.0, "noise_amplitude": 200, "seed": 8},
    ]
    assert [row["audio_path"] for row in augmented_rows] == [
        str(tmp_path / "augmented" / "utt-001_aug_quiet.wav"),
        str(tmp_path / "augmented" / "utt-001_aug_noisy.wav"),
    ]
