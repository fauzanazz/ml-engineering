from pathlib import Path

import pandas as pd

from house_prices.data import load_train_validation_split


def test_split_uses_chronological_sale_columns(tmp_path: Path):
    data_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5],
            "YrSold": [2009, 2006, 2010, 2008, 2007],
            "MoSold": [1, 1, 1, 1, 1],
            "SalePrice": [1, 2, 3, 4, 5],
        }
    ).to_csv(data_path, index=False)

    train_features, val_features, _, _, strategy = load_train_validation_split(data_path, val_size=0.4)

    assert strategy == "chronological_by_YrSold_MoSold"
    assert train_features["YrSold"].max() <= val_features["YrSold"].min()


def test_split_falls_back_to_random_without_chronology(tmp_path: Path):
    data_path = tmp_path / "train.csv"
    pd.DataFrame({"Id": range(10), "LotArea": range(10), "SalePrice": range(10)}).to_csv(data_path, index=False)

    *_, strategy = load_train_validation_split(data_path, val_size=0.2, random_state=1)

    assert strategy == "random_holdout_no_reliable_chronology_columns"
