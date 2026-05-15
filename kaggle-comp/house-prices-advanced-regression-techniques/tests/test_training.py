from pathlib import Path

import pandas as pd

from house_prices.models import RandomForestFactory
from house_prices.training import train_model


def test_train_model_returns_metrics(tmp_path: Path):
    data_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Id": range(1, 11),
            "LotArea": [8000, 9000, 8500, 9100, 10000, 7500, 8200, 8800, 9300, 9700],
            "Neighborhood": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
            "SalePrice": [200000, 210000, 205000, 220000, 240000, 190000, 202000, 215000, 235000, 225000],
        }
    ).to_csv(data_path, index=False)

    result = train_model(
        data_path,
        RandomForestFactory(random_state=1),
        val_size=0.3,
        random_state=1,
    )

    assert len(result.predictions) == 3
    assert result.metrics.rmse >= 0
