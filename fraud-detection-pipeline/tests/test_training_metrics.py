import pandas as pd

from fraud_detection.metrics import ClassificationMetrics
from fraud_detection.models import LightGbmFactory
from fraud_detection.training import train_one_batch


def _write_csv(tmp_path, rows: dict) -> object:
    path = tmp_path / "creditcard.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_training_result_includes_classification_metrics(tmp_path):
    data_path = _write_csv(
        tmp_path,
        {
            "Time": [6, 1, 5, 2, 4, 3],
            "V1": [4.2, 0.0, 4.1, 0.1, 4.0, 0.2],
            "V2": [4.2, 0.0, 4.1, 0.1, 4.0, 0.2],
            "Amount": [102.0, 1.0, 101.0, 1.1, 100.0, 1.2],
            "Class": [1, 0, 1, 0, 1, 0],
        },
    )

    result = train_one_batch(
        data_path=data_path,
        model_factory=LightGbmFactory(),
        batch_size=6,
        test_size=0.5,
    )

    assert isinstance(result.metrics, ClassificationMetrics)
    assert 0.0 <= result.metrics.precision <= 1.0
    assert 0.0 <= result.metrics.recall <= 1.0
    assert 0.0 <= result.metrics.f1 <= 1.0
    assert 0.0 <= result.metrics.pr_auc <= 1.0
