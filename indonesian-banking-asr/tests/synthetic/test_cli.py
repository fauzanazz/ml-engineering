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
