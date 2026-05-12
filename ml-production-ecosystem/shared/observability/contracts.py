"""Observability contracts shared across learning stages."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class MetricPoint:
    """Single metric observation emitted by ML system components."""

    name: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)


class ObservabilitySink(Protocol):
    """Boundary for emitting metrics and operational signals."""

    def emit_metric(self, metric: MetricPoint) -> None:
        """Emit metric point to configured observability backend."""
