import json
import subprocess
import sys
import wave


def test_cli_generates_pilot_manifest_jsonl(tmp_path):
    output_path = tmp_path / "manifest.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.synthetic.cli",
            "--output-path",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["template_id"] == "check_balance_001"
    assert rows[0]["intent"] == "check_balance"
    assert rows[0]["entities"]


def test_cli_generates_audio_manifest_jsonl(tmp_path):
    input_path = tmp_path / "text_manifest.jsonl"
    audio_manifest_path = tmp_path / "audio_manifest.jsonl"
    audio_dir = tmp_path / "audio"
    input_path.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "text": "Saya mau cek saldo rekening 1234567890.",
            }
        )
        + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.synthetic.cli",
            "tts",
            "--input-path",
            str(input_path),
            "--output-path",
            str(audio_manifest_path),
            "--audio-dir",
            str(audio_dir),
            "--sample-rate",
            "8000",
            "--duration-sec",
            "0.25",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in audio_manifest_path.read_text().splitlines()]
    assert rows[0]["audio_path"] == str(audio_dir / "utt-001.wav")
    assert rows[0]["duration_sec"] == 0.25
    assert rows[0]["sample_rate"] == 8000
    with wave.open(rows[0]["audio_path"], "rb") as wav_file:
        assert wav_file.getframerate() == 8000
        assert wav_file.getnchannels() == 1


def test_cli_validates_audio_manifest_jsonl(tmp_path):
    input_path = tmp_path / "text_manifest.jsonl"
    audio_manifest_path = tmp_path / "audio_manifest.jsonl"
    qa_report_path = tmp_path / "audio_qa.jsonl"
    audio_dir = tmp_path / "audio"
    input_path.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "text": "Saya mau cek saldo rekening 1234567890.",
            }
        )
        + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.synthetic.cli",
            "tts",
            "--input-path",
            str(input_path),
            "--output-path",
            str(audio_manifest_path),
            "--audio-dir",
            str(audio_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.synthetic.cli",
            "audio-qa",
            "--input-path",
            str(audio_manifest_path),
            "--output-path",
            str(qa_report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    reports = [json.loads(line) for line in qa_report_path.read_text().splitlines()]
    assert reports == [
        {
            "checked_rows": 1,
            "valid_rows": 1,
            "invalid_rows": 0,
            "errors": [],
        }
    ]


def test_cli_generates_augmented_audio_manifest_jsonl(tmp_path):
    input_path = tmp_path / "text_manifest.jsonl"
    audio_manifest_path = tmp_path / "audio_manifest.jsonl"
    augmented_manifest_path = tmp_path / "augmented_manifest.jsonl"
    audio_dir = tmp_path / "audio"
    augmented_dir = tmp_path / "augmented"
    input_path.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "text": "Saya mau cek saldo rekening 1234567890.",
            }
        )
        + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.synthetic.cli",
            "tts",
            "--input-path",
            str(input_path),
            "--output-path",
            str(audio_manifest_path),
            "--audio-dir",
            str(audio_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.synthetic.cli",
            "augment-audio",
            "--input-path",
            str(audio_manifest_path),
            "--output-path",
            str(augmented_manifest_path),
            "--output-dir",
            str(augmented_dir),
            "--gain",
            "0.5",
            "--noise-amplitude",
            "200",
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in augmented_manifest_path.read_text().splitlines()]
    assert rows[0]["audio_path"] == str(augmented_dir / "utt-001_gain-0.5.wav")
    assert rows[0]["source_audio_path"] == str(audio_dir / "utt-001.wav")
    assert rows[0]["augmentation"] == {
        "gain": 0.5,
        "noise_amplitude": 200,
        "seed": 7,
    }


def test_cli_generates_multiple_augmented_audio_profiles(tmp_path):
    input_path = tmp_path / "text_manifest.jsonl"
    audio_manifest_path = tmp_path / "audio_manifest.jsonl"
    augmented_manifest_path = tmp_path / "augmented_manifest.jsonl"
    audio_dir = tmp_path / "audio"
    augmented_dir = tmp_path / "augmented"
    input_path.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "text": "Saya mau cek saldo rekening 1234567890.",
            }
        )
        + "\n"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.synthetic.cli",
            "tts",
            "--input-path",
            str(input_path),
            "--output-path",
            str(audio_manifest_path),
            "--audio-dir",
            str(audio_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.synthetic.cli",
            "augment-audio",
            "--input-path",
            str(audio_manifest_path),
            "--output-path",
            str(augmented_manifest_path),
            "--output-dir",
            str(augmented_dir),
            "--profile",
            "quiet:0.5:0:7",
            "--profile",
            "noisy:1.0:200:8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in augmented_manifest_path.read_text().splitlines()]
    assert [row["utterance_id"] for row in rows] == [
        "utt-001_aug_quiet",
        "utt-001_aug_noisy",
    ]
    assert [row["augmentation"] for row in rows] == [
        {"name": "quiet", "gain": 0.5, "noise_amplitude": 0, "seed": 7},
        {"name": "noisy", "gain": 1.0, "noise_amplitude": 200, "seed": 8},
    ]


def test_cli_merges_clean_and_augmented_audio_manifests(tmp_path):
    clean_manifest_path = tmp_path / "audio_manifest.jsonl"
    augmented_manifest_path = tmp_path / "augmented_manifest.jsonl"
    dataset_manifest_path = tmp_path / "dataset_manifest.jsonl"
    clean_manifest_path.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "audio_path": "audio/utt-001.wav",
                "text": "Saya mau cek saldo.",
            }
        )
        + "\n"
    )
    augmented_manifest_path.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001_aug_noisy",
                "audio_path": "augmented/utt-001_aug_noisy.wav",
                "source_audio_path": "audio/utt-001.wav",
                "text": "Saya mau cek saldo.",
                "augmentation": {"name": "noisy", "gain": 1.0, "noise_amplitude": 200, "seed": 7},
            }
        )
        + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.synthetic.cli",
            "merge-audio-manifests",
            "--clean-input-path",
            str(clean_manifest_path),
            "--augmented-input-path",
            str(augmented_manifest_path),
            "--output-path",
            str(dataset_manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in dataset_manifest_path.read_text().splitlines()]
    assert [row["dataset_variant"] for row in rows] == ["clean", "augmented"]
