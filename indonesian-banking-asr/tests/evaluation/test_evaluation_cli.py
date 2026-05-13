import json
import subprocess
import sys


def test_evaluation_cli_writes_wer_and_entity_summary(tmp_path):
    manifest_path = tmp_path / "dataset_manifest.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "evaluation_summary.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "text": "rekening 1234567890 sebesar dua juta rupiah",
                "entities": [
                    {"type": "ACCOUNT_NUMBER", "text": "1234567890"},
                    {"type": "AMOUNT", "text": "dua juta rupiah"},
                ],
            }
        )
        + "\n"
    )
    predictions_path.write_text(
        json.dumps(
            {
                "utterance_id": "utt-001",
                "hypothesis": "rekening 1234567890 sebesar dua ratus rupiah",
            }
        )
        + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "indonesian_banking_asr.evaluation.cli",
            "--manifest-path",
            str(manifest_path),
            "--predictions-path",
            str(predictions_path),
            "--output-path",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [
        {
            "rows": 1,
            "word_errors": 1,
            "reference_words": 6,
            "wer": 1 / 6,
            "entity_errors": 1,
            "entities": 2,
            "entity_error_rate": 0.5,
            "errors_by_type": {"AMOUNT": 1},
        }
    ]
