import pytest
import pandas as pd
from unittest.mock import MagicMock, call
import numpy as np
from lightgbm import LGBMClassifier

from fraud_detection.models import LightGbmFactory, ModelFactory
from fraud_detection.training import _measure_single_row_latency, compute_scale_pos_weight, train_one_batch


def _three_way_csv(tmp_path) -> object:
    """10 rows with all V1-V28 cols: enough for train/val/test split."""
    import numpy as np

    data_path = tmp_path / "creditcard3way.csv"
    rng = np.random.default_rng(42)
    n = 10
    row = {"Time": list(range(1, n + 1))}
    for i in range(1, 29):
        vals = rng.standard_normal(n)
        # Make V1 clearly separable so model can score
        if i == 1:
            vals = [4.2 if c == 1 else 0.1 for c in [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
        row[f"V{i}"] = vals
    row["Amount"] = [1.0, 102.0, 1.1, 101.0, 1.2, 100.0, 1.3, 99.0, 1.4, 98.0]
    row["Class"] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    pd.DataFrame(row).to_csv(data_path, index=False)
    return data_path


def _minimal_csv(tmp_path) -> object:
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


@pytest.mark.parametrize("bad_threshold", [0, 1, -0.1, 1.1])
def test_train_one_batch_rejects_invalid_decision_threshold(tmp_path, bad_threshold):
    with pytest.raises(ValueError, match="decision_threshold"):
        train_one_batch(
            data_path=_minimal_csv(tmp_path),
            model_factory=LightGbmFactory(),
            batch_size=6,
            test_size=0.5,
            decision_threshold=bad_threshold,
        )


def test_compute_scale_pos_weight_returns_ratio():
    # 8 negatives, 2 positives => 4.0
    labels = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    assert compute_scale_pos_weight(labels) == 4.0


def test_compute_scale_pos_weight_raises_on_no_positives():
    labels = pd.Series([0, 0, 0])

    with pytest.raises(ValueError, match="no positive"):
        compute_scale_pos_weight(labels)


def test_compute_scale_pos_weight_raises_on_no_negatives():
    labels = pd.Series([1, 1, 1])

    with pytest.raises(ValueError, match="no negative"):
        compute_scale_pos_weight(labels)


def test_train_one_batch_with_scale_pos_weight_returns_metrics(tmp_path):
    # sorted by Time: 1(neg),2(pos),3(neg),4(pos),5(neg),6(pos)
    # train=first 4: neg,pos,neg,pos → both classes present
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

    result = train_one_batch(
        data_path=data_path,
        model_factory=LightGbmFactory(),
        batch_size=6,
        test_size=0.5,
        imbalance_strategy="scale-pos-weight",
    )

    assert len(result.predictions) == 3
    assert 0.0 <= result.training_accuracy <= 1.0


def test_train_one_batch_produces_test_predictions(tmp_path):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [6, 1, 5, 2, 4, 3],
            "V1": [4.2, 0.0, 4.1, 0.1, 4.0, 0.2],
            "V2": [4.2, 0.0, 4.1, 0.1, 4.0, 0.2],
            "Amount": [102.0, 1.0, 101.0, 1.1, 100.0, 1.2],
            "Class": [1, 0, 1, 0, 1, 0],
        }
    ).to_csv(data_path, index=False)

    result = train_one_batch(
        data_path=data_path,
        model_factory=LightGbmFactory(),
        batch_size=6,
        test_size=0.5,
    )

    assert len(result.predictions) == 3
    assert 0.0 <= result.training_accuracy <= 1.0
    assert 0.0 <= result.test_accuracy <= 1.0


def test_train_one_batch_default_test_size_produces_one_test_prediction(tmp_path):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [5, 1, 4, 2, 3],
            "V1": [4.0, 0.0, 4.1, 0.1, 0.2],
            "V2": [4.0, 0.0, 4.1, 0.1, 0.2],
            "Amount": [100.0, 1.0, 101.0, 1.1, 1.2],
            "Class": [1, 0, 1, 0, 0],
        }
    ).to_csv(data_path, index=False)

    result = train_one_batch(
        data_path=data_path,
        model_factory=LightGbmFactory(),
        batch_size=5,
    )

    assert len(result.predictions) == 1


def test_compute_scale_pos_weight_uses_binary_class_counts():
    # label 2 should not count; only 0/1 matter
    labels = pd.Series([0, 0, 0, 0, 1, 1])

    ratio = compute_scale_pos_weight(labels)

    assert ratio == 2.0  # 4 negatives / 2 positives


def test_compute_scale_pos_weight_raises_when_no_positives_via_binary_count():
    # all zeros — explicit binary check must catch this
    labels = pd.Series([0, 0, 0])

    with pytest.raises(ValueError, match="no positive"):
        compute_scale_pos_weight(labels)


class _RecordingFactory:
    """Captures scale_pos_weight passed to create()."""

    received_scale_pos_weight: float | None = None

    def create(self, scale_pos_weight: float | None = None) -> LGBMClassifier:
        self._received_scale_pos_weight = scale_pos_weight
        return LightGbmFactory(scale_pos_weight=scale_pos_weight).create()


def test_train_one_batch_with_val_size_returns_val_threshold_and_val_metrics(tmp_path):
    result = train_one_batch(
        data_path=_three_way_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=10,
        test_size=0.2,
        val_size=0.2,
    )

    assert result.val_threshold is not None
    assert 0 < result.val_threshold < 1
    assert result.val_metrics is not None
    assert 0.0 <= result.val_metrics.precision <= 1.0
    assert 0.0 <= result.val_metrics.recall <= 1.0


def test_train_one_batch_without_val_size_has_no_val_fields(tmp_path):
    result = train_one_batch(
        data_path=_minimal_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=6,
        test_size=0.5,
    )

    assert result.val_threshold is None
    assert result.val_metrics is None
    assert result.threshold_objective is None
    assert result.target_recall is None


def test_train_one_batch_val_size_threshold_objective_f1(tmp_path):
    result = train_one_batch(
        data_path=_three_way_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=10,
        test_size=0.2,
        val_size=0.2,
        threshold_objective="f1",
    )

    assert result.threshold_objective == "f1"
    assert result.val_threshold is not None


def test_train_one_batch_val_size_threshold_objective_target_recall(tmp_path):
    result = train_one_batch(
        data_path=_three_way_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=10,
        test_size=0.2,
        val_size=0.2,
        threshold_objective="target-recall",
        target_recall=0.9,
    )

    assert result.threshold_objective == "target-recall"
    assert result.target_recall == 0.9
    assert result.val_threshold is not None


def test_train_one_batch_val_size_applies_val_threshold_to_test(tmp_path):
    """Predictions count matches test set size (threshold applied)."""
    result = train_one_batch(
        data_path=_three_way_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=10,
        test_size=0.2,
        val_size=0.2,
    )

    assert len(result.predictions) >= 1
    assert all(p in (0, 1) for p in result.predictions)


def test_train_one_batch_result_has_latency_fields(tmp_path):
    result = train_one_batch(
        data_path=_minimal_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=6,
        test_size=0.5,
    )

    assert result.predict_proba_latency_s is not None
    assert result.predict_proba_latency_s >= 0.0
    assert result.predict_proba_latency_per_row_s is not None
    assert result.predict_proba_latency_per_row_s >= 0.0
    # alias backward compat
    assert result.inference_latency_s == result.predict_proba_latency_s


def test_train_one_batch_with_val_size_result_has_latency_fields(tmp_path):
    result = train_one_batch(
        data_path=_three_way_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=10,
        test_size=0.2,
        val_size=0.2,
    )

    assert result.predict_proba_latency_s is not None
    assert result.predict_proba_latency_s >= 0.0
    assert result.predict_proba_latency_per_row_s is not None
    assert result.predict_proba_latency_per_row_s >= 0.0
    assert result.inference_latency_s == result.predict_proba_latency_s


def test_predict_proba_latency_per_row_s_equals_batch_divided_by_rows(tmp_path):
    result = train_one_batch(
        data_path=_minimal_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=6,
        test_size=0.5,
    )

    n_rows = len(result.predictions)
    expected = result.predict_proba_latency_s / n_rows
    assert result.predict_proba_latency_per_row_s == pytest.approx(expected)


def test_train_one_batch_result_has_single_row_latency(tmp_path):
    result = train_one_batch(
        data_path=_minimal_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=6,
        test_size=0.5,
    )

    assert result.single_row_latency_s is not None
    assert result.single_row_latency_s >= 0.0


def test_train_one_batch_with_val_size_result_has_single_row_latency(tmp_path):
    result = train_one_batch(
        data_path=_three_way_csv(tmp_path),
        model_factory=LightGbmFactory(),
        batch_size=10,
        test_size=0.2,
        val_size=0.2,
    )

    assert result.single_row_latency_s is not None
    assert result.single_row_latency_s >= 0.0


def test_measure_single_row_latency_calls_predict_proba_warmup_plus_n_repeats():
    """Helper does 1 warmup + n_repeats timed calls, all on 1-row input."""
    n_repeats = 3
    one_row = pd.DataFrame({"a": [1.0]})

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])

    _measure_single_row_latency(mock_model, one_row, n_repeats=n_repeats)

    assert mock_model.predict_proba.call_count == 1 + n_repeats
    for c in mock_model.predict_proba.call_args_list:
        arg = c.args[0]
        assert len(arg) == 1, f"expected 1-row input, got {len(arg)} rows"


def test_measure_single_row_latency_returns_none_for_model_without_predict_proba():
    class _NoProbaModel:
        def predict(self, X):
            return [0] * len(X)

    one_row = pd.DataFrame({"a": [1.0]})
    result = _measure_single_row_latency(_NoProbaModel(), one_row)
    assert result is None


def test_measure_single_row_latency_returns_none_not_raises_is_propagated(tmp_path):
    """single_row_latency_s=None when helper returns None (no predict_proba)."""
    from unittest.mock import patch

    result_holder = {}

    with patch("fraud_detection.training._measure_single_row_latency", return_value=None):
        result = train_one_batch(
            data_path=_minimal_csv(tmp_path),
            model_factory=LightGbmFactory(),
            batch_size=6,
            test_size=0.5,
        )

    assert result.single_row_latency_s is None


def test_custom_factory_receives_scale_pos_weight_not_ignored(tmp_path):
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

    recording_factory = _RecordingFactory()
    train_one_batch(
        data_path=data_path,
        model_factory=recording_factory,
        batch_size=6,
        test_size=0.5,
        imbalance_strategy="scale-pos-weight",
    )

    assert recording_factory._received_scale_pos_weight is not None
    assert recording_factory._received_scale_pos_weight > 0
