from pathlib import Path

import pandas as pd

DEFAULT_TARGET_COLUMN = "SalePrice"
DEFAULT_ID_COLUMN = "Id"
DATE_COLUMNS = ("YrSold", "MoSold")


def load_training_data(path: Path | str, target_column: str = DEFAULT_TARGET_COLUMN):
    data = pd.read_csv(path)
    target = data[target_column]
    features = data.drop(columns=[target_column])
    return features, target


def has_chronology_columns(features: pd.DataFrame) -> bool:
    return all(column in features.columns for column in DATE_COLUMNS)


def _chronological_indices(features: pd.DataFrame, val_size: float) -> tuple[pd.Index, pd.Index]:
    sort_columns = [*DATE_COLUMNS]
    if DEFAULT_ID_COLUMN in features.columns:
        sort_columns.append(DEFAULT_ID_COLUMN)

    ordered_index = features.sort_values(sort_columns, kind="mergesort").index
    val_count = max(1, int(round(len(ordered_index) * val_size)))
    if val_count >= len(ordered_index):
        raise ValueError("validation split leaves no training rows")
    return ordered_index[:-val_count], ordered_index[-val_count:]


def _random_indices(features: pd.DataFrame, target: pd.Series, val_size: float, random_state: int):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        features.index,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )


def load_train_validation_split(
    path: Path | str,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    val_size: float = 0.2,
    random_state: int = 42,
):
    if not 0 < val_size < 1:
        raise ValueError("val_size must satisfy 0 < val_size < 1")

    features, target = load_training_data(path, target_column=target_column)
    if has_chronology_columns(features):
        train_index, val_index = _chronological_indices(features, val_size)
        split_strategy = "chronological_by_YrSold_MoSold"
    else:
        train_index, val_index = _random_indices(features, target, val_size, random_state)
        split_strategy = "random_holdout_no_reliable_chronology_columns"

    return (
        features.loc[train_index],
        features.loc[val_index],
        target.loc[train_index],
        target.loc[val_index],
        split_strategy,
    )


def load_kaggle_test_data(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)
