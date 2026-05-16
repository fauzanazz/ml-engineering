from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import copy
import json
from threading import Lock
from typing import TypeVar

T = TypeVar("T")


def stable_cache_key(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


class PredictionCache:
    def __init__(self, max_entries: int = 100) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, object] = OrderedDict()
        self._hit_count = 0
        self._miss_count = 0
        self._lock = Lock()

    def get_or_compute(self, payload: dict[str, object], predict: Callable[[], T]) -> T:
        key = stable_cache_key(payload)
        with self._lock:
            if key in self._entries:
                self._hit_count += 1
                self._entries.move_to_end(key)
                return copy.deepcopy(self._entries[key])
            self._miss_count += 1

        result = predict()

        with self._lock:
            self._entries[key] = copy.deepcopy(result)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return result

    def summary(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0.0
            return {
                "cache": {
                    "hit_count": self._hit_count,
                    "miss_count": self._miss_count,
                    "entry_count": len(self._entries),
                    "max_entries": self.max_entries,
                    "hit_rate": round(hit_rate, 6),
                }
            }
