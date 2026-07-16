import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.cli import main as train_main
from fraud_detection.score_cli import main as score_main
from fraud_detection.score_cli import parse_args


def _run(tmp_path):
    data_path = tmp_path / "data.csv"
    raw = pd.DataFrame(
        {
            "Time": np.arange(20, dtype=float),
            "Amount": np.resize([1.0, 100.0], 20),
            "V1": np.resize([0.0, 4.0], 20),
            "Class": np.resize([0, 1], 20),
        }
    )
    raw.to_csv(data_path, index=False)
    artifact_dir = tmp_path / "runs"
    train_main(
        [
            "--data-path",
            str(data_path),
            "--batch-size",
            "20",
            "--test-size",
            "0.2",
            "--model",
            "logistic-regression",
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    return next(artifact_dir.iterdir()), raw.drop(columns=["Class"]).iloc[-1].to_dict()


def test_score_cli_prints_one_exact_json_object(tmp_path, capsys):
    run_dir, transaction = _run(tmp_path)
    capsys.readouterr()
    input_path = tmp_path / "transaction.json"
    input_path.write_text(json.dumps(transaction))

    score_main(
        [
            "--run-dir",
            str(run_dir),
            "--input-path",
            str(input_path),
            "--trust-artifact",
        ]
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert set(payload) == {
        "fraud_probability",
        "decision_threshold",
        "prediction",
    }
    assert isinstance(payload["fraud_probability"], float)
    assert isinstance(payload["decision_threshold"], float)
    assert type(payload["prediction"]) is int
    assert output.count("\n") == 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json", "error:"),
        ("[]", "error: input JSON must be one object"),
        ('{"Time": 1}', "error: missing input columns"),
    ],
)
def test_score_cli_reports_input_errors(tmp_path, content, message):
    run_dir, _ = _run(tmp_path)
    input_path = tmp_path / "bad.json"
    input_path.write_text(content)

    with pytest.raises(SystemExit, match=message):
        score_main(
            [
                "--run-dir",
                str(run_dir),
                "--input-path",
                str(input_path),
                "--trust-artifact",
            ]
        )


def test_score_cli_requires_explicit_trust_flag():
    with pytest.raises(SystemExit):
        parse_args(["--run-dir", "run", "--input-path", "transaction.json"])
