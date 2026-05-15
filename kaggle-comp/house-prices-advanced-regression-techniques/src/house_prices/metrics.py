from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    log_rmse: float
    r2: float


def _log_rmse(labels: np.ndarray, predictions: np.ndarray) -> float:
    clipped_predictions = np.maximum(predictions, np.finfo(float).eps)
    return float(np.sqrt(mean_squared_error(np.log(labels), np.log(clipped_predictions))))


class SklearnRegressionMetrics:
    def compute(self, labels: np.ndarray, predictions: np.ndarray) -> RegressionMetrics:
        return RegressionMetrics(
            mae=float(mean_absolute_error(labels, predictions)),
            rmse=float(np.sqrt(mean_squared_error(labels, predictions))),
            log_rmse=_log_rmse(labels, predictions),
            r2=float(r2_score(labels, predictions)),
        )
