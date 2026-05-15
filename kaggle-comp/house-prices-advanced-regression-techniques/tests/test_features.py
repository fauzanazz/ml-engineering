import pandas as pd
import pytest

from house_prices.features import FeaturePipeline, add_house_price_features


def test_feature_pipeline_engineers_missing_and_unknown_categories():
    train = pd.DataFrame(
        {
            "Id": [1, 2],
            "TotalBsmtSF": [500, 600],
            "1stFlrSF": [700, 800],
            "2ndFlrSF": [100, 200],
            "YrSold": [2010, 2010],
            "YearBuilt": [2000, 1990],
            "Neighborhood": ["CollgCr", "Veenker"],
            "Alley": [None, "Grvl"],
            "ExterQual": ["Gd", "TA"],
        }
    )
    test = train.copy()
    test.loc[0, "Neighborhood"] = "NewPlace"

    transformed = FeaturePipeline().fit(train).transform(test)

    assert "Id" not in transformed.columns
    assert "TotalSF" in transformed.columns
    assert transformed.isna().sum().sum() == 0


def test_feature_engineering_adds_house_age():
    features = pd.DataFrame({"YrSold": [2010], "YearBuilt": [2000]})

    engineered = add_house_price_features(features)

    assert engineered.loc[0, "HouseAgeAtSale"] == 10


def test_feature_pipeline_blocks_target_leakage_columns():
    features = pd.DataFrame({"LotArea": [1, 2], "SalePrice": [3, 4]})

    with pytest.raises(ValueError, match="Potential target leakage"):
        FeaturePipeline().fit(features)
