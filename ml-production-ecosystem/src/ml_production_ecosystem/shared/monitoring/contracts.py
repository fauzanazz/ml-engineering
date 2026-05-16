"""Monitoring contracts shared across learning stages."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MonitoringResult:
    """Result for one monitoring check."""

    check_name: str
    passed: bool
    message: str


class MonitoringCheck(Protocol):
    """Boundary for data, model, or service monitoring checks."""

    def run(self) -> MonitoringResult:
        """Run check and return monitoring result."""
