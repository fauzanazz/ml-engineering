import pytest

from scale_reliability.cache import PredictionCache, stable_cache_key


def test_prediction_cache_records_miss_then_hit_for_equivalent_payload() -> None:
    cache = PredictionCache(max_entries=10)
    calls = 0

    def predict() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"recommendations": [1, 2]}

    first = cache.get_or_compute({"top_k": 2, "user_id": 10}, predict)
    second = cache.get_or_compute({"user_id": 10, "top_k": 2}, predict)

    assert first == {"recommendations": [1, 2]}
    assert second == first
    assert calls == 1
    assert cache.summary() == {
        "cache": {
            "hit_count": 1,
            "miss_count": 1,
            "entry_count": 1,
            "max_entries": 10,
            "hit_rate": 0.5,
        }
    }


def test_prediction_cache_records_miss_for_different_payload() -> None:
    cache = PredictionCache(max_entries=10)

    cache.get_or_compute({"user_id": 10, "top_k": 2}, lambda: {"recommendations": [1, 2]})
    cache.get_or_compute({"user_id": 10, "top_k": 3}, lambda: {"recommendations": [1, 2, 3]})

    assert cache.summary()["cache"]["hit_count"] == 0
    assert cache.summary()["cache"]["miss_count"] == 2
    assert cache.summary()["cache"]["entry_count"] == 2


def test_prediction_cache_does_not_cache_failed_prediction() -> None:
    cache = PredictionCache(max_entries=10)
    attempts = 0

    def fail_once_then_succeed() -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("prediction failed")
        return {"recommendations": [1]}

    with pytest.raises(RuntimeError, match="prediction failed"):
        cache.get_or_compute({"user_id": 10}, fail_once_then_succeed)

    result = cache.get_or_compute({"user_id": 10}, fail_once_then_succeed)

    assert result == {"recommendations": [1]}
    assert attempts == 2
    assert cache.summary()["cache"]["miss_count"] == 2
    assert cache.summary()["cache"]["hit_count"] == 0
    assert cache.summary()["cache"]["entry_count"] == 1


def test_stable_cache_key_is_deterministic_for_equivalent_payloads() -> None:
    assert stable_cache_key({"top_k": 2, "user_id": 10}) == stable_cache_key({"user_id": 10, "top_k": 2})


def test_prediction_cache_stays_bounded_by_max_entries() -> None:
    cache = PredictionCache(max_entries=2)

    cache.get_or_compute({"request": 1}, lambda: {"value": 1})
    cache.get_or_compute({"request": 2}, lambda: {"value": 2})
    cache.get_or_compute({"request": 3}, lambda: {"value": 3})

    assert cache.summary()["cache"]["entry_count"] == 2
    assert cache.summary()["cache"]["max_entries"] == 2


def test_prediction_cache_rejects_invalid_max_entries() -> None:
    with pytest.raises(ValueError, match="max_entries must be at least 1"):
        PredictionCache(max_entries=0)
