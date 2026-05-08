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
    split_index = int(len(batch) * (1 - test_size))
    train = batch.iloc[:split_index]
    test = batch.iloc[split_index:]
    return (
        train.drop(columns=[target_column]),
        test.drop(columns=[target_column]),
        train[target_column],
        test[target_column],
    )
