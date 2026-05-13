import json
import subprocess
import sys


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
