from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from threading import Lock
from typing import TypeVar

T = TypeVar("T")


class BackpressureRejected(Exception):
    def __init__(self, max_in_flight: int) -> None:
        self.status_code = 429
        self.response = {
            "status": "rejected",
            "reason": "max_in_flight_reached",
            "max_in_flight": max_in_flight,
        }
        super().__init__(self.response["reason"])


class InFlightLimiter:
    def __init__(self, max_in_flight: int) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        self.max_in_flight = max_in_flight
        self._active_count = 0
        self._lock = Lock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_count

    @contextmanager
    def slot(self):
        self.acquire()
        try:
            yield
        finally:
            self.release()

    def run(self, work: Callable[[], T]) -> T:
        with self.slot():
            return work()

    def acquire(self) -> None:
        with self._lock:
            if self._active_count >= self.max_in_flight:
                raise BackpressureRejected(self.max_in_flight)
            self._active_count += 1

    def release(self) -> None:
        with self._lock:
            if self._active_count > 0:
                self._active_count -= 1
