import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import FeaturePipeline


def _make_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v_cols = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    return pd.DataFrame(
        {
            "Time": rng.integers(0, 86400, n).astype(float),
            "Amount": rng.exponential(scale=100, size=n),
            **v_cols,
        }
    )


def _make_df_with_zero_amount(n: int) -> pd.DataFrame:
    df = _make_df(n)
    df.loc[0, "Amount"] = 0.0
    return df


@pytest.fixture()
def train_df():
    return _make_df(100, seed=1)


@pytest.fixture()
def val_df():
    return _make_df(30, seed=2)


@pytest.fixture()
def fitted_pipeline(train_df):
    pipeline = FeaturePipeline()
    pipeline.fit(train_df)
    return pipeline


# --- RED: watch these fail before implementation exists ---


class TestEngineeredFeatures:
    def test_log_amount_equals_log1p_of_amount(self, fitted_pipeline, train_df):
        result = fitted_pipeline.transform(train_df)
        expected = np.log1p(train_df["Amount"])
        np.testing.assert_allclose(
            result["log_amount_raw"].values, expected.values, rtol=1e-6
        )

    def test_amount_is_zero_flag(self, fitted_pipeline):
        df = _make_df_with_zero_amount(10)
        result = fitted_pipeline.transform(df)
        assert result["amount_is_zero"].iloc[0] == 1
        assert result["amount_is_zero"].iloc[1:].sum() == 0

    def test_hour_of_day_range(self, fitted_pipeline, train_df):
        result = fitted_pipeline.transform(train_df)
        assert result["hour_of_day"].between(0, 23).all()

    def test_day_is_integer_multiple_of_86400(self, fitted_pipeline):
        df = pd.DataFrame(
            {
                "Time": [0.0, 86400.0, 172800.0, 43200.0],
                "Amount": [10.0, 20.0, 30.0, 5.0],
                **{f"V{i}": [0.0] * 4 for i in range(1, 29)},
            }
        )
        pipeline = FeaturePipeline()
        pipeline.fit(df)
        result = pipeline.transform(df)
        assert result["day"].tolist() == [0, 1, 2, 0]

    def test_is_night_true_between_midnight_and_6am(self, fitted_pipeline):
        hours = list(range(24))
        times = [h * 3600.0 for h in hours]
        df = pd.DataFrame(
            {
                "Time": times,
                "Amount": [10.0] * 24,
                **{f"V{i}": [0.0] * 24 for i in range(1, 29)},
            }
        )
        pipeline = FeaturePipeline()
        pipeline.fit(df)
        result = pipeline.transform(df)
        for h in range(24):
            expected = int(h < 6)
            assert result["is_night"].iloc[h] == expected, f"hour={h} failed"


class TestScaling:
    def test_scaled_columns_present(self, fitted_pipeline, train_df):
        result = fitted_pipeline.transform(train_df)
        assert "log_amount_scaled" in result.columns
        assert "amount_scaled" in result.columns

    def test_train_scaled_mean_near_zero(self, fitted_pipeline, train_df):
        result = fitted_pipeline.transform(train_df)
        # RobustScaler: median → ~0, not strict mean=0; just check plausible range
        assert abs(result["log_amount_scaled"].median()) < 1.0
        assert abs(result["amount_scaled"].median()) < 1.0

    def test_val_transform_uses_train_stats_not_val_stats(self, train_df, val_df):
        """Val median of scaled col != 0 proves train scaler used, not refitted on val."""
        pipeline = FeaturePipeline()
        pipeline.fit(train_df)

        train_result = pipeline.transform(train_df)
        val_result = pipeline.transform(val_df)

        # Scaler fitted on train: train median ~0, val median != train median
        train_median = train_result["log_amount_scaled"].median()
        val_median = val_result["log_amount_scaled"].median()

        # If pipeline refitted on val, val_median would also be ~0.
        # With train-fitted scaler, val_median differs.
        # Seeds 1 vs 2 produce different distributions → gap > 0.01
        assert abs(val_median - train_median) > 0.01

    def test_transform_without_fit_raises(self, train_df):
        pipeline = FeaturePipeline()
        with pytest.raises(RuntimeError, match="fit"):
            pipeline.transform(train_df)


class TestPassthrough:
    def test_v_columns_preserved_unchanged(self, fitted_pipeline, train_df):
        result = fitted_pipeline.transform(train_df)
        for i in range(1, 29):
            col = f"V{i}"
            np.testing.assert_array_equal(result[col].values, train_df[col].values)

    def test_output_row_count_matches_input(self, fitted_pipeline, train_df, val_df):
        assert len(fitted_pipeline.transform(train_df)) == len(train_df)
        assert len(fitted_pipeline.transform(val_df)) == len(val_df)

    def test_class_column_not_required_or_produced(self, fitted_pipeline, train_df):
        result = fitted_pipeline.transform(train_df)
        assert "Class" not in result.columns

    def test_original_amount_and_time_not_in_output(self, fitted_pipeline, train_df):
        result = fitted_pipeline.transform(train_df)
        assert "Amount" not in result.columns
        assert "Time" not in result.columns
