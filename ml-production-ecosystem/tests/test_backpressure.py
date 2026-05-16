import pytest

from ml_production_ecosystem.scale_reliability.backpressure import InFlightLimiter, BackpressureRejected


def test_limiter_allows_work_below_max_in_flight() -> None:
    limiter = InFlightLimiter(max_in_flight=2)

    result = limiter.run(lambda: "ok")

    assert result == "ok"
    assert limiter.active_count == 0


def test_limiter_rejects_when_max_in_flight_reached() -> None:
    limiter = InFlightLimiter(max_in_flight=1)

    with limiter.slot():
        with pytest.raises(BackpressureRejected) as error:
            limiter.run(lambda: "overflow")

    assert error.value.response == {
        "status": "rejected",
        "reason": "max_in_flight_reached",
        "max_in_flight": 1,
    }
    assert error.value.status_code == 429
    assert limiter.active_count == 0


def test_limiter_releases_slot_after_handled_failure_result() -> None:
    limiter = InFlightLimiter(max_in_flight=1)

    result = limiter.run(lambda: {"status": "failed"})

    assert result == {"status": "failed"}
    assert limiter.active_count == 0
    assert limiter.run(lambda: "next") == "next"


def test_limiter_releases_slot_after_exception() -> None:
    limiter = InFlightLimiter(max_in_flight=1)

    def fail() -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        limiter.run(fail)

    assert limiter.active_count == 0
    assert limiter.run(lambda: "recovered") == "recovered"


def test_limiter_rejects_invalid_max_in_flight() -> None:
    with pytest.raises(ValueError, match="max_in_flight must be at least 1"):
        InFlightLimiter(max_in_flight=0)
