import pytest

from fraud_detection.cli import parse_args


def test_parse_args_default_decision_threshold_is_0_5():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.decision_threshold == pytest.approx(0.5)


def test_parse_args_accepts_custom_decision_threshold():
    args = parse_args(
        ["--data-path", "data/creditcard.csv", "--decision-threshold", "0.2"]
    )

    assert args.decision_threshold == pytest.approx(0.2)


@pytest.mark.parametrize("bad", ["0", "1", "-0.1", "1.1"])
def test_parse_args_rejects_invalid_decision_threshold(bad):
    with pytest.raises(SystemExit):
        parse_args(
            ["--data-path", "data/creditcard.csv", "--decision-threshold", bad]
        )


def test_parse_args_default_threshold_sweep_is_false():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.threshold_sweep is False


def test_parse_args_accepts_threshold_sweep_flag():
    args = parse_args(["--data-path", "data/creditcard.csv", "--threshold-sweep"])

    assert args.threshold_sweep is True
