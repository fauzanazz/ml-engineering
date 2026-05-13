from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class RateLimiter:
    seconds_per_request: float
    sleep: Callable[[float], None] = time.sleep
    _has_waited: bool = False

    def wait_before_request(self) -> None:
        if not self._has_waited:
            self.sleep(0.0)
            self._has_waited = True
            return
        self.sleep(self.seconds_per_request)
