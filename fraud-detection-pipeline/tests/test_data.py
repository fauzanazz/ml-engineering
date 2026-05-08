import pandas as pd
import pytest

from fraud_detection.data import load_batch, load_time_split_batch


def test_load_batch_returns_features_and_target(tmp_path):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "V1": [0.1, 0.2, 0.3],
            "Amount": [10.0, 20.0, 30.0],
            "Class": [0, 1, 0],
        }
    ).to_csv(data_path, index=False)

    features, target = load_batch(data_path, batch_size=2)

    assert list(features.columns) == ["V1", "Amount"]
    assert target.tolist() == [0, 1]


def test_load_time_split_batch_sorts_by_time_before_split(tmp_path):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [50, 10, 40, 20, 30],
            "V1": [5, 1, 4, 2, 3],
            "Class": [1, 0, 1, 0, 0],
        }
    ).to_csv(data_path, index=False)

    train_features, test_features, train_target, test_target = load_time_split_batch(
        data_path,
        batch_size=5,
        test_size=0.2,
    )

    assert train_features["Time"].tolist() == [10, 20, 30, 40]
    assert test_features["Time"].tolist() == [50]
    assert train_target.tolist() == [0, 0, 0, 1]
    assert test_target.tolist() == [1]


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1])
def test_load_time_split_batch_rejects_invalid_test_size(tmp_path, test_size):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [1, 2],
            "V1": [0.1, 0.2],
            "Class": [0, 1],
        }
    ).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="test_size must satisfy 0 < test_size < 1"):
        load_time_split_batch(data_path, batch_size=2, test_size=test_size)
