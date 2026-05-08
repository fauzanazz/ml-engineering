import pytest

from fraud_detection.cli import parse_args


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
