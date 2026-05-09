import pandas as pd
import pytest

from fraud_detection.data import load_batch, load_three_way_split, load_time_split_batch


def test_load_batch_returns_features_and_target(tmp_path):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "V1": [0.1, 0.2, 0.3],
            "Amount": [10.0, 20.0, 30.0],
            "Class": [0, 1, 0],
        }
    ).to_csv(data_path, index=False)

    features, target = load_batch(data_path, batch_size=2)

    assert list(features.columns) == ["V1", "Amount"]
    assert target.tolist() == [0, 1]


def test_load_time_split_batch_sorts_by_time_before_split(tmp_path):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [50, 10, 40, 20, 30],
            "V1": [5, 1, 4, 2, 3],
            "Class": [1, 0, 1, 0, 0],
        }
    ).to_csv(data_path, index=False)

    train_features, test_features, train_target, test_target = load_time_split_batch(
        data_path,
        batch_size=5,
        test_size=0.2,
    )

    assert train_features["Time"].tolist() == [10, 20, 30, 40]
    assert test_features["Time"].tolist() == [50]
    assert train_target.tolist() == [0, 0, 0, 1]
    assert test_target.tolist() == [1]
    # indexes reset — positional alignment holds
    assert train_features.index.tolist() == list(range(len(train_features)))
    assert test_features.index.tolist() == list(range(len(test_features)))


def test_load_time_split_batch_drops_duplicates_within_each_split(tmp_path):
    """Duplicate = same (Time, V1) feature row; second occurrence dropped within split."""
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [10, 10, 20, 30, 30],
            "V1": [1, 1, 2, 3, 3],
            "Class": [0, 0, 1, 1, 1],
        }
    ).to_csv(data_path, index=False)

    train_features, test_features, train_target, test_target = load_time_split_batch(
        data_path,
        batch_size=5,
        test_size=0.4,
    )

    # train: rows at Time=10(dup→1), Time=20 → 2 rows
    # test:  rows at Time=30(dup→1) → 1 row
    assert train_features["Time"].tolist() == [10, 20]
    assert train_features["V1"].tolist() == [1, 2]
    assert len(train_features) == len(train_target) == 2

    assert test_features["Time"].tolist() == [30]
    assert test_features["V1"].tolist() == [3]
    assert len(test_features) == len(test_target) == 1


def test_load_time_split_batch_removes_cross_split_leakage(tmp_path):
    """Test row with same (Time, V1) as any train row is dropped to prevent leakage."""
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [10, 20, 20, 30],
            "V1": [1, 2, 2, 3],
            "Class": [0, 1, 1, 0],
        }
    ).to_csv(data_path, index=False)
    # After sort+split (test_size=0.4, 4 rows → split=2):
    #   train: Time=10/V1=1, Time=20/V1=2
    #   test (before dedup): Time=20/V1=2 (dup of train), Time=30/V1=3
    # Expected test after leak removal: only Time=30/V1=3

    train_features, test_features, train_target, test_target = load_time_split_batch(
        data_path,
        batch_size=4,
        test_size=0.4,
    )

    assert test_features["Time"].tolist() == [30]
    assert test_features["V1"].tolist() == [3]
    assert test_target.tolist() == [0]

    assert len(test_features) == len(test_target) == 1
    assert len(train_features) == len(train_target) == 2

    train_tuples = set(map(tuple, train_features.values.tolist()))
    test_tuples = set(map(tuple, test_features.values.tolist()))
    assert train_tuples.isdisjoint(test_tuples)


def test_load_time_split_batch_target_alignment_and_no_feature_overlap(tmp_path):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [10, 20, 30, 40, 50],
            "V1": [1, 2, 3, 4, 5],
            "Class": [0, 1, 0, 1, 0],
        }
    ).to_csv(data_path, index=False)

    train_features, test_features, train_target, test_target = load_time_split_batch(
        data_path,
        batch_size=5,
        test_size=0.4,
    )

    # All rows unique — train 3 rows, test 2 rows, no overlap
    assert len(train_features) == len(train_target) == 3
    assert len(test_features) == len(test_target) == 2

    # positional alignment via reset index
    assert train_features.index.tolist() == [0, 1, 2]
    assert test_features.index.tolist() == [0, 1]

    train_tuples = set(map(tuple, train_features.values.tolist()))
    test_tuples = set(map(tuple, test_features.values.tolist()))
    assert train_tuples.isdisjoint(test_tuples)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1])
def test_load_time_split_batch_rejects_invalid_test_size(tmp_path, test_size):
    data_path = tmp_path / "creditcard.csv"
    pd.DataFrame(
        {
            "Time": [1, 2],
            "V1": [0.1, 0.2],
            "Class": [0, 1],
        }
    ).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="test_size must satisfy 0 < test_size < 1"):
        load_time_split_batch(data_path, batch_size=2, test_size=test_size)


# ── load_three_way_split ──────────────────────────────────────────────────────


def _make_csv(tmp_path, rows: dict) -> object:
    path = tmp_path / "creditcard.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_three_way_split_chronological_order(tmp_path):
    """Train < val < test in Time after sort."""
    path = _make_csv(
        tmp_path,
        {
            "Time": [50, 10, 40, 20, 30, 60, 70, 80, 90, 100],
            "V1": list(range(10)),
            "Class": [0] * 10,
        },
    )

    train_f, val_f, test_f, _, _, _ = load_three_way_split(
        path, val_size=0.2, test_size=0.2
    )

    assert train_f["Time"].max() <= val_f["Time"].min()
    assert val_f["Time"].max() <= test_f["Time"].min()


def test_three_way_split_sizes(tmp_path):
    """10 rows, val_size=0.2, test_size=0.2 → train=6, val=2, test=2."""
    path = _make_csv(
        tmp_path,
        {
            "Time": list(range(10)),
            "V1": list(range(10)),
            "Class": [0] * 10,
        },
    )

    train_f, val_f, test_f, train_t, val_t, test_t = load_three_way_split(
        path, val_size=0.2, test_size=0.2
    )

    assert len(train_f) == len(train_t) == 6
    assert len(val_f) == len(val_t) == 2
    assert len(test_f) == len(test_t) == 2


def test_three_way_split_within_split_dedup(tmp_path):
    """Duplicate feature rows (incl. Time) dropped within each split."""
    path = _make_csv(
        tmp_path,
        {
            # 9 rows; duplicate pairs at Time=10,20,70
            "Time": [10, 10, 20, 30, 40, 50, 60, 70, 70],
            "V1":   [ 1,  1,  2,  3,  4,  5,  6,  7,  7],
            "Class":[  0,  0,  1,  0,  1,  0,  1,  0,  0],
        },
    )
    # val_size=0.2, test_size=0.2 → split indices on 9 rows:
    #   train end = int(9*0.6)=5, val end = int(9*0.8)=7
    #   train: Time 10,10,20,30,40 → after dedup: 10,20,30,40 (4)
    #   val:   Time 50,60          → no dups        (2)
    #   test:  Time 70,70          → after dedup: 70 (1)

    train_f, val_f, test_f, train_t, val_t, test_t = load_three_way_split(
        path, val_size=0.2, test_size=0.2
    )

    assert len(train_f) == len(train_t) == 4
    assert len(val_f) == len(val_t) == 2
    assert len(test_f) == len(test_t) == 1


def test_three_way_split_no_cross_split_overlap(tmp_path):
    """No feature row appears in more than one split after dedup + leak removal."""
    path = _make_csv(
        tmp_path,
        {
            # row at Time=40,V1=4 appears in both val and test positions
            "Time": [10, 20, 30, 40, 40, 50, 60, 70, 80, 100],
            "V1":   [ 1,  2,  3,  4,  4,  5,  6,  7,  8,   9],
            "Class":[  0,  1,  0,  1,  1,  0,  1,  0,  1,   0],
        },
    )

    train_f, val_f, test_f, _, _, _ = load_three_way_split(
        path, val_size=0.2, test_size=0.2
    )

    train_hashes = set(pd.util.hash_pandas_object(train_f, index=False))
    val_hashes = set(pd.util.hash_pandas_object(val_f, index=False))
    test_hashes = set(pd.util.hash_pandas_object(test_f, index=False))

    assert train_hashes.isdisjoint(val_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert val_hashes.isdisjoint(test_hashes)


def test_three_way_split_target_alignment(tmp_path):
    """Features and target align positionally after reset_index."""
    path = _make_csv(
        tmp_path,
        {
            "Time":  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "V1":    [ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10],
            "Class": [ 0,  1,  0,  1,  0,  1,  0,  1,  0,   1],
        },
    )

    train_f, val_f, test_f, train_t, val_t, test_t = load_three_way_split(
        path, val_size=0.2, test_size=0.2
    )

    for feat, tgt in [(train_f, train_t), (val_f, val_t), (test_f, test_t)]:
        assert feat.index.tolist() == list(range(len(feat)))
        assert tgt.index.tolist() == list(range(len(tgt)))
        assert len(feat) == len(tgt)


@pytest.mark.parametrize(
    "val_size,test_size",
    [
        (0, 0.2),
        (0.2, 0),
        (1, 0.2),
        (0.2, 1),
        (-0.1, 0.2),
        (0.5, 0.6),   # val+test >= 1
    ],
)
def test_three_way_split_rejects_invalid_sizes(tmp_path, val_size, test_size):
    path = _make_csv(
        tmp_path,
        {"Time": [1, 2, 3], "V1": [0.1, 0.2, 0.3], "Class": [0, 1, 0]},
    )

    with pytest.raises(ValueError):
        load_three_way_split(path, val_size=val_size, test_size=test_size)
