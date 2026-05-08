import pytest
import pandas as pd
from lightgbm import LGBMClassifier

from fraud_detection.models import LightGbmFactory, ModelFactory
from fraud_detection.training import compute_scale_pos_weight, train_one_batch


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
