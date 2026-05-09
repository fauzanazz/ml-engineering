import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

_ALL_V_COLS = [f"V{i}" for i in range(1, 29)]
_REQUIRED_COLS = ["Time", "Amount"]
_SCALE_COLS = ["log_amount_raw", "Amount"]
_SCALE_OUTPUT = ["log_amount_scaled", "amount_scaled"]


def _validate_schema(df: pd.DataFrame, required_v_cols: list[str] | None = None) -> None:
    missing = [col for col in _REQUIRED_COLS if col not in df.columns]
    if required_v_cols:
        missing += [col for col in required_v_cols if col not in df.columns]
    if missing:
        raise ValueError(f"FeaturePipeline: missing required columns: {missing}")


def _engineer(df: pd.DataFrame, v_cols: list[str]) -> pd.DataFrame:
    hour = (df["Time"] % 86400 // 3600).astype(int)
    day = (df["Time"] // 86400).astype(int)
    return pd.DataFrame(
        {
            **{col: df[col] for col in v_cols},
            "log_amount_raw": np.log1p(df["Amount"]),
            "Amount": df["Amount"],
            "amount_is_zero": (df["Amount"] == 0).astype(int),
            "hour_of_day": hour,
            "day": day,
            "is_night": (hour < 6).astype(int),
        }
    )


class FeaturePipeline:
    def __init__(self) -> None:
        self._scaler: RobustScaler | None = None
        self._v_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        _validate_schema(df)
        self._v_cols = [col for col in _ALL_V_COLS if col in df.columns]
        engineered = _engineer(df, self._v_cols)
        self._scaler = RobustScaler()
        self._scaler.fit(engineered[_SCALE_COLS])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None:
            raise RuntimeError("Pipeline not fitted. Call fit() before transform().")
        _validate_schema(df, required_v_cols=self._v_cols)

        engineered = _engineer(df, self._v_cols)
        scaled = self._scaler.transform(engineered[_SCALE_COLS])

        result = engineered.drop(columns=["Amount"]).copy()
        for i, col in enumerate(_SCALE_OUTPUT):
            result[col] = scaled[:, i]

        return result
