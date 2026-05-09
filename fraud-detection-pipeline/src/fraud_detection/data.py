import numpy as np
import pandas as pd


def load_batch(path, batch_size: int, target_column: str = "Class"):
    batch = pd.read_csv(path, nrows=batch_size)
    target = batch[target_column]
    features = batch.drop(columns=[target_column])
    return features, target


def load_time_split_batch(path, batch_size: int, target_column: str = "Class", test_size: float = 0.2):
    if not 0 < test_size < 1:
        raise ValueError("test_size must satisfy 0 < test_size < 1")

    batch = pd.read_csv(path, nrows=batch_size).sort_values("Time")
    feature_cols = [c for c in batch.columns if c != target_column]

    split_index = int(len(batch) * (1 - test_size))
    train = batch.iloc[:split_index].drop_duplicates(subset=feature_cols)
    test = batch.iloc[split_index:].drop_duplicates(subset=feature_cols)

    # Remove test rows whose feature signature matches any train row (cross-split leakage).
    train_hash = pd.util.hash_pandas_object(train[feature_cols], index=False).to_numpy(dtype=np.uint64)
    test_hash = pd.util.hash_pandas_object(test[feature_cols], index=False).to_numpy(dtype=np.uint64)
    test_no_leak = test[~np.isin(test_hash, train_hash)]

    train_features = train[feature_cols].reset_index(drop=True)
    test_features = test_no_leak[feature_cols].reset_index(drop=True)
    train_target = train[target_column].reset_index(drop=True)
    test_target = test_no_leak[target_column].reset_index(drop=True)

    return train_features, test_features, train_target, test_target


def load_three_way_split(
    path,
    val_size: float = 0.2,
    test_size: float = 0.2,
    target_column: str = "Class",
    batch_size: int | None = None,
):
    if not (0 < val_size < 1) or not (0 < test_size < 1) or val_size + test_size >= 1:
        raise ValueError(
            "val_size and test_size must each be in (0, 1) and sum to less than 1"
        )

    df = pd.read_csv(path, nrows=batch_size).sort_values("Time")
    feature_cols = [c for c in df.columns if c != target_column]

    n = len(df)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    raw_train = df.iloc[:train_end]
    raw_val = df.iloc[train_end:val_end]
    raw_test = df.iloc[val_end:]

    train = raw_train.drop_duplicates(subset=feature_cols)
    val = raw_val.drop_duplicates(subset=feature_cols)
    test = raw_test.drop_duplicates(subset=feature_cols)

    train_hashes = pd.util.hash_pandas_object(train[feature_cols], index=False).to_numpy(dtype=np.uint64)

    val_row_hashes = pd.util.hash_pandas_object(val[feature_cols], index=False).to_numpy(dtype=np.uint64)
    val = val[~np.isin(val_row_hashes, train_hashes)]

    val_hashes = pd.util.hash_pandas_object(val[feature_cols], index=False).to_numpy(dtype=np.uint64)
    prior_hashes = np.concatenate([train_hashes, val_hashes])
    test_row_hashes = pd.util.hash_pandas_object(test[feature_cols], index=False).to_numpy(dtype=np.uint64)
    test = test[~np.isin(test_row_hashes, prior_hashes)]

    return (
        train[feature_cols].reset_index(drop=True),
        val[feature_cols].reset_index(drop=True),
        test[feature_cols].reset_index(drop=True),
        train[target_column].reset_index(drop=True),
        val[target_column].reset_index(drop=True),
        test[target_column].reset_index(drop=True),
    )
