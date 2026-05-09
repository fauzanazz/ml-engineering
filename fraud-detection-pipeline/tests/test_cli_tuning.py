import pandas as pd
import pytest

from fraud_detection.cli import parse_args


def _minimal_csv(tmp_path):
    """12 balanced rows so train split (--test-size 0.5 → 6 rows) has >=2 per class."""
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": list(range(1, 13)),
            "V1": [0.0, 4.2, 0.1, 4.1, 0.2, 4.0, 0.3, 4.3, 0.4, 4.4, 0.5, 4.5],
            "V2": [0.0, 4.2, 0.1, 4.1, 0.2, 4.0, 0.3, 4.3, 0.4, 4.4, 0.5, 4.5],
            "Amount": [1.0, 102.0, 1.1, 101.0, 1.2, 100.0, 1.3, 103.0, 1.4, 104.0, 1.5, 105.0],
            "Class": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    ).to_csv(data_path, index=False)
    return data_path


def test_parse_args_tune_default_is_false():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.tune is False


def test_parse_args_tune_flag_sets_true():
    args = parse_args(["--data-path", "data/creditcard.csv", "--tune", "--model", "random-forest"])

    assert args.tune is True


def test_parse_args_tune_n_candidates_default():
    args = parse_args(["--data-path", "data/creditcard.csv"])

    assert args.tune_n_candidates == 10


def test_parse_args_tune_n_candidates_custom():
    args = parse_args(["--data-path", "data/creditcard.csv", "--tune-n-candidates", "5"])

    assert args.tune_n_candidates == 5


def test_parse_args_tune_n_iter_alias_maps_to_tune_n_candidates():
    """--tune-n-iter is a legacy alias; must write to tune_n_candidates dest."""
    args = parse_args(["--data-path", "data/creditcard.csv", "--tune-n-iter", "7"])

    assert args.tune_n_candidates == 7


def test_parse_args_tune_rejects_non_tunable_model():
    with pytest.raises(SystemExit):
        parse_args([
            "--data-path", "data/creditcard.csv",
            "--tune",
            "--model", "logistic-regression",
        ])


def test_parse_args_tune_accepts_random_forest():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--tune",
        "--model", "random-forest",
    ])

    assert args.tune is True
    assert args.model == "random-forest"


def test_parse_args_tune_accepts_xgboost():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--tune",
        "--model", "xgboost",
    ])

    assert args.tune is True
    assert args.model == "xgboost"


def test_main_tune_prints_tuning_best_score_and_params(tmp_path, capsys):
    from fraud_detection.cli import main

    main([
        "--data-path", str(_minimal_csv(tmp_path)),
        "--batch-size", "12",
        "--test-size", "0.5",
        "--no-artifacts",
        "--tune",
        "--tune-n-candidates", "2",
        "--model", "random-forest",
    ])

    captured = capsys.readouterr()
    assert "tuning_best_score=" in captured.out
    assert "tuning_best_params=" in captured.out


def test_main_tune_xgboost_prints_tuning_fields(tmp_path, capsys):
    from fraud_detection.cli import main

    main([
        "--data-path", str(_minimal_csv(tmp_path)),
        "--batch-size", "12",
        "--test-size", "0.5",
        "--no-artifacts",
        "--tune",
        "--tune-n-candidates", "2",
        "--model", "xgboost",
    ])

    captured = capsys.readouterr()
    assert "tuning_best_score=" in captured.out
    assert "tuning_best_params=" in captured.out


def test_main_tune_then_trains_prints_accuracy(tmp_path, capsys):
    """After tuning, training still runs and prints accuracy."""
    from fraud_detection.cli import main

    main([
        "--data-path", str(_minimal_csv(tmp_path)),
        "--batch-size", "12",
        "--test-size", "0.5",
        "--no-artifacts",
        "--tune",
        "--tune-n-candidates", "2",
        "--model", "random-forest",
    ])

    captured = capsys.readouterr()
    assert "training_accuracy=" in captured.out
    assert "test_accuracy=" in captured.out


# ---------------------------------------------------------------------------
# Wave 3 blockers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1", "-10"])
def test_parse_args_rejects_tune_n_candidates_less_than_1(bad):
    with pytest.raises(SystemExit):
        parse_args([
            "--data-path", "data/creditcard.csv",
            "--tune-n-candidates", bad,
        ])


def test_parse_args_accepts_tune_n_candidates_of_1():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--tune-n-candidates", "1",
    ])
    assert args.tune_n_candidates == 1


def _three_way_csv(tmp_path):
    """20 rows with balanced classes — enough for three-way split + CV."""
    import pandas as pd
    data_path = tmp_path / "creditcard_3way.csv"
    n = 20
    pd.DataFrame(
        {
            "Time": list(range(1, n + 1)),
            "V1": [float(i % 2) * 4.0 for i in range(n)],
            "V2": [float((i + 1) % 2) * 2.0 for i in range(n)],
            "Amount": [1.0 + i for i in range(n)],
            "Class": [i % 2 for i in range(n)],
        }
    ).to_csv(data_path, index=False)
    return data_path


def test_main_tune_with_val_size_uses_training_split(tmp_path, capsys):
    """--tune + --val-size tunes on train split only, still prints tuning fields."""
    from fraud_detection.cli import main

    main([
        "--data-path", str(_three_way_csv(tmp_path)),
        "--batch-size", "20",
        "--test-size", "0.2",
        "--val-size", "0.2",
        "--no-artifacts",
        "--tune",
        "--tune-n-candidates", "2",
        "--model", "random-forest",
    ])

    captured = capsys.readouterr()
    assert "tuning_best_score=" in captured.out
    assert "training_accuracy=" in captured.out


# ---------------------------------------------------------------------------
# lightgbm tuning via CLI
# ---------------------------------------------------------------------------


def test_parse_args_tune_accepts_lightgbm():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--tune",
        "--model", "lightgbm",
    ])

    assert args.tune is True
    assert args.model == "lightgbm"


def test_main_tune_lightgbm_prints_tuning_fields(tmp_path, capsys):
    from fraud_detection.cli import main

    main([
        "--data-path", str(_minimal_csv(tmp_path)),
        "--batch-size", "12",
        "--test-size", "0.5",
        "--no-artifacts",
        "--tune",
        "--tune-n-candidates", "1",
        "--model", "lightgbm",
    ])

    captured = capsys.readouterr()
    assert "tuning_best_score=" in captured.out
    assert "tuning_best_params=" in captured.out


def test_main_tune_lightgbm_then_trains(tmp_path, capsys):
    from fraud_detection.cli import main

    main([
        "--data-path", str(_minimal_csv(tmp_path)),
        "--batch-size", "12",
        "--test-size", "0.5",
        "--no-artifacts",
        "--tune",
        "--tune-n-candidates", "1",
        "--model", "lightgbm",
    ])

    captured = capsys.readouterr()
    assert "training_accuracy=" in captured.out
    assert "test_accuracy=" in captured.out


# ---------------------------------------------------------------------------
# Final review issues
# ---------------------------------------------------------------------------


def test_tune_help_text_describes_adaptive_search():
    """--tune help must mention 'adaptive hyperparameter search', not 'AutoML'."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.suppress(SystemExit):
        with contextlib.redirect_stdout(buf):
            parse_args(["--data-path", "x", "--help"])
    help_text = buf.getvalue()
    assert "adaptive hyperparameter search" in help_text
    assert "AutoML" not in help_text


def test_tune_help_shows_tune_n_candidates_as_primary():
    """--tune-n-candidates must appear in help; --tune-n-iter must not (suppressed)."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.suppress(SystemExit):
        with contextlib.redirect_stdout(buf):
            parse_args(["--data-path", "x", "--help"])
    help_text = buf.getvalue()
    assert "--tune-n-candidates" in help_text
    assert "--tune-n-iter" not in help_text


@pytest.mark.parametrize("bad", ["501", "1000", "5000"])
def test_parse_args_rejects_tune_n_candidates_above_500(bad):
    with pytest.raises(SystemExit):
        parse_args([
            "--data-path", "data/creditcard.csv",
            "--tune-n-candidates", bad,
        ])


def test_parse_args_accepts_tune_n_candidates_of_500():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--tune-n-candidates", "500",
    ])
    assert args.tune_n_candidates == 500


def test_parse_args_accepts_tune_n_candidates_of_200():
    args = parse_args([
        "--data-path", "data/creditcard.csv",
        "--tune-n-candidates", "200",
    ])
    assert args.tune_n_candidates == 200


def test_main_tune_raises_system_exit_with_context_on_value_error(tmp_path, monkeypatch):
    """ValueError from tuning must become SystemExit raised from exc (preserves traceback)."""
    from fraud_detection.cli import main
    from fraud_detection import tuning

    def _raise(*args, **kwargs):
        raise ValueError("injected tuning failure")

    monkeypatch.setattr(tuning, "tune_random_forest", _raise)

    with pytest.raises(SystemExit) as exc_info:
        main([
            "--data-path", str(_minimal_csv(tmp_path)),
            "--batch-size", "12",
            "--test-size", "0.5",
            "--no-artifacts",
            "--tune",
            "--tune-n-candidates", "2",
            "--model", "random-forest",
        ])

    # SystemExit raised *from* ValueError — __cause__ must be set
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ValueError)
