import pandas as pd
import pytest

from fraud_detection.cli import main, parse_args


def _minimal_csv(tmp_path):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [1, 2, 3, 4, 5, 6],
            "V1": [0.0, 4.2, 0.1, 4.1, 0.2, 4.0],
            "V2": [0.0, 4.2, 0.1, 4.1, 0.2, 4.0],
            "Amount": [1.0, 102.0, 1.1, 101.0, 1.2, 100.0],
            "Class": [0, 1, 0, 1, 0, 1],
        }
    ).to_csv(data_path, index=False)
    return data_path


def test_main_prints_latency_fields(tmp_path, capsys):
    main([
        "--data-path", str(_minimal_csv(tmp_path)),
        "--batch-size", "6",
        "--test-size", "0.5",
        "--no-artifacts",
    ])

    captured = capsys.readouterr()
    assert "predict_proba_latency_s=" in captured.out
    assert "predict_proba_latency_per_row_s=" in captured.out


def test_parse_args_reads_training_options():
    args = parse_args(
        [
            "--data-path",
            "data/creditcard.csv",
            "--batch-size",
            "128",
            "--target-column",
            "Class",
            "--test-size",
            "0.3",
        ]
    )

    assert str(args.data_path) == "data/creditcard.csv"
    assert args.batch_size == 128
    assert args.target_column == "Class"
    assert args.test_size == 0.3


def test_parse_args_uses_training_defaults():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.batch_size == 256
    assert args.target_column == "Class"
    assert args.test_size == 0.2


@pytest.mark.parametrize("test_size", ["0", "1", "-0.1", "1.1"])
def test_parse_args_rejects_invalid_test_size(test_size):
    with pytest.raises(SystemExit):
        parse_args(["--data-path", "data/creditcard.csv", "--test-size", test_size])


def test_parse_args_default_imbalance_strategy_is_none():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.imbalance_strategy == "none"


def test_parse_args_accepts_scale_pos_weight_strategy():
    args = parse_args(
        ["--data-path", "data/creditcard.csv", "--imbalance-strategy", "scale-pos-weight"]
    )

    assert args.imbalance_strategy == "scale-pos-weight"


def test_parse_args_rejects_unknown_imbalance_strategy():
    with pytest.raises(SystemExit):
        parse_args(["--data-path", "data/creditcard.csv", "--imbalance-strategy", "smote"])


def test_parse_args_default_val_size_is_none():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.val_size is None


def test_parse_args_accepts_val_size():
    args = parse_args(["--data-path", "data/creditcard.csv", "--val-size", "0.2"])

    assert args.val_size == pytest.approx(0.2)


@pytest.mark.parametrize("bad", ["0", "1", "-0.1", "1.1"])
def test_parse_args_rejects_invalid_val_size(bad):
    with pytest.raises(SystemExit):
        parse_args(["--data-path", "data/creditcard.csv", "--val-size", bad])


def test_parse_args_default_threshold_objective_is_f1():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.threshold_objective == "f1"


def test_parse_args_accepts_threshold_objective_target_recall():
    args = parse_args(
        ["--data-path", "data/creditcard.csv", "--threshold-objective", "target-recall", "--val-size", "0.2"]
    )

    assert args.threshold_objective == "target-recall"


def test_parse_args_rejects_unknown_threshold_objective():
    with pytest.raises(SystemExit):
        parse_args(
            ["--data-path", "data/creditcard.csv", "--threshold-objective", "precision"]
        )


def test_parse_args_default_target_recall_is_none():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.target_recall is None


def test_parse_args_accepts_target_recall():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--threshold-objective", "target-recall",
        "--val-size", "0.2",
        "--target-recall", "0.95",
    ])

    assert args.target_recall == pytest.approx(0.95)


def test_parse_args_default_model_is_lightgbm():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.model == "lightgbm"


@pytest.mark.parametrize("model", ["lightgbm", "logistic-regression", "decision-tree", "random-forest", "xgboost"])
def test_parse_args_accepts_all_model_choices(model):
    args = parse_args(["--data-path", "data/creditcard.csv", "--model", model])

    assert args.model == model


def test_parse_args_rejects_unknown_model():
    with pytest.raises(SystemExit):
        parse_args(["--data-path", "data/creditcard.csv", "--model", "svm"])


def test_parse_args_rejects_custom_decision_threshold_when_val_size_set():
    with pytest.raises(SystemExit):
        parse_args([
            "--data-path", "data/creditcard.csv",
            "--val-size", "0.2",
            "--decision-threshold", "0.3",
        ])


def test_parse_args_accepts_default_decision_threshold_with_val_size():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--val-size", "0.2",
        "--decision-threshold", "0.5",
    ])
    assert args.decision_threshold == pytest.approx(0.5)
    assert args.val_size == pytest.approx(0.2)


def test_parse_args_allows_custom_threshold_without_val_size():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--decision-threshold", "0.3",
    ])
    assert args.decision_threshold == pytest.approx(0.3)
