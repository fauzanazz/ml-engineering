import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_LIKE_COLUMNS = {"SalePrice", "SalePriceLog", "SalePrice_log", "target"}
DROP_COLUMNS = {"Id"}
NONE_MEANS_MISSING = {
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
}
QUALITY_COLUMNS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
QUALITY_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
TARGET_ENCODED_COLUMNS = [
    "Neighborhood", "MSZoning", "ExterQual", "BsmtQual", "KitchenQual", "MasVnrType",
    "BsmtFinType1", "FireplaceQu", "BsmtExposure", "BsmtCond", "GarageFinish", "Alley",
]


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_house_price_features(features: pd.DataFrame) -> pd.DataFrame:
    engineered = features.copy()
    for column in NONE_MEANS_MISSING:
        if column in engineered.columns:
            engineered[column] = engineered[column].fillna("None")

    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(engineered.columns):
        engineered["TotalSF"] = engineered["TotalBsmtSF"] + engineered["1stFlrSF"] + engineered["2ndFlrSF"]
    if {"BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"}.issubset(engineered.columns):
        engineered["TotalBath"] = engineered["FullBath"] + 0.5 * engineered["HalfBath"] + engineered["BsmtFullBath"] + 0.5 * engineered["BsmtHalfBath"]
    if {"YrSold", "YearBuilt"}.issubset(engineered.columns):
        engineered["HouseAgeAtSale"] = (engineered["YrSold"] - engineered["YearBuilt"]).clip(lower=0)
    if {"YrSold", "YearRemodAdd"}.issubset(engineered.columns):
        engineered["YearsSinceRemodel"] = (engineered["YrSold"] - engineered["YearRemodAdd"]).clip(lower=0)
    if {"GarageYrBlt", "YrSold"}.issubset(engineered.columns):
        engineered["GarageAgeAtSale"] = (engineered["YrSold"] - engineered["GarageYrBlt"]).clip(lower=0)
    if {"OverallQual", "GrLivArea"}.issubset(engineered.columns):
        engineered["QualityLivingArea"] = engineered["OverallQual"] * engineered["GrLivArea"]

    quality_score_columns = []
    for column in QUALITY_COLUMNS:
        if column in engineered.columns:
            score_column = f"{column}Score"
            engineered[score_column] = engineered[column].map(QUALITY_MAP).fillna(0)
            quality_score_columns.append(score_column)
    if quality_score_columns:
        engineered["TotalQualityScore"] = engineered[quality_score_columns].sum(axis=1)

    return engineered


def assert_no_leakage_columns(features: pd.DataFrame) -> None:
    leaked_columns = sorted(TARGET_LIKE_COLUMNS.intersection(features.columns))
    if leaked_columns:
        raise ValueError(f"Potential target leakage columns present: {leaked_columns}")


class FeaturePipeline:
    def __init__(self) -> None:
        self._transformer: ColumnTransformer | None = None
        self._feature_columns: list[str] = []
        self._target_encoding_global_mean: float | None = None
        self._target_encoding_maps: dict[str, pd.Series] = {}

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> "FeaturePipeline":
        assert_no_leakage_columns(features)
        engineered = add_house_price_features(features.drop(columns=list(DROP_COLUMNS.intersection(features.columns))))
        engineered = self._fit_target_encoding(engineered, target)
        self._feature_columns = list(engineered.columns)
        numeric_columns = engineered.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = [column for column in self._feature_columns if column not in numeric_columns]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", _make_one_hot_encoder()),
            ]
        )
        self._transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self._transformer.fit(engineered)
        return self

    def _fit_target_encoding(self, features: pd.DataFrame, target: pd.Series | None) -> pd.DataFrame:
        if target is None:
            return features

        encoded = features.copy()
        log_target = np.log(pd.Series(target, index=features.index).astype(float))
        self._target_encoding_global_mean = float(log_target.mean())
        self._target_encoding_maps = {}
        for column in TARGET_ENCODED_COLUMNS:
            if column not in encoded.columns:
                continue

            grouped = pd.DataFrame({"key": encoded[column], "target": log_target}).groupby("key")["target"].agg(["mean", "count"])
            smoothed = (grouped["mean"] * grouped["count"] + self._target_encoding_global_mean * 10) / (grouped["count"] + 10)
            self._target_encoding_maps[column] = smoothed
            encoded[f"TargetMean_{column}"] = encoded[column].map(smoothed).fillna(self._target_encoding_global_mean)
        return encoded

    def _transform_target_encoding(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._target_encoding_global_mean is None:
            return features

        encoded = features.copy()
        for column, mapping in self._target_encoding_maps.items():
            encoded[f"TargetMean_{column}"] = encoded[column].map(mapping).fillna(self._target_encoding_global_mean)
        return encoded

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._transformer is None:
            raise RuntimeError("FeaturePipeline must be fitted before transform")

        assert_no_leakage_columns(features)
        cleaned = features.drop(columns=list(DROP_COLUMNS.intersection(features.columns)))
        engineered = self._transform_target_encoding(add_house_price_features(cleaned)).reindex(columns=self._feature_columns)
        values = self._transformer.transform(engineered)
        columns = self._transformer.get_feature_names_out()
        return pd.DataFrame(values, columns=columns, index=features.index)

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)
