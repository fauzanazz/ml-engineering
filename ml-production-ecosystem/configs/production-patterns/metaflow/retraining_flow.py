"""Metaflow-style flow skeleton for production retraining orchestration.

This file is import-safe without Metaflow installed. Placeholder decorators keep
step names and command examples readable until a real Metaflow runtime is added.
"""

from __future__ import annotations

import subprocess

SCHEDULED_RETRAIN_COMMAND = (
    "uv",
    "run",
    "production-scheduled-retrain",
    "--config",
    "configs/foundation-recommender.yaml",
    "--set-active",
    "--require-quality-gate",
    "--output-path",
    "artifacts/reports/production-patterns/scheduled-retraining.json",
)

MONITOR_AFTER_RETRAIN_COMMAND = (
    "uv",
    "run",
    "production-monitor",
    "--base-url",
    "http://127.0.0.1:8000",
    "--max-error-count",
    "0",
    "--max-drift-score",
    "0.2",
    "--max-latency-ms-last",
    "100",
)

VALIDATE_CONFIG_COMMAND = ("test", "-f", "configs/foundation-recommender.yaml")

try:
    from metaflow import FlowSpec, schedule, step  # type: ignore[import-not-found]
except ImportError:

    class FlowSpec:  # type: ignore[no-redef]
        def next(self, _next_step: object) -> None:
            return None

    def schedule(*_args: object, **_kwargs: object):  # type: ignore[no-redef]
        def decorator(cls: type[FlowSpec]) -> type[FlowSpec]:
            return cls

        return decorator

    def step(func):  # type: ignore[no-redef]
        return func


def run_command(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True)


@schedule(cron="0 2 * * *")
class ProductionRetrainingFlow(FlowSpec):
    @step
    def start(self) -> None:
        run_command(VALIDATE_CONFIG_COMMAND)
        self.next(self.scheduled_retrain)

    @step
    def scheduled_retrain(self) -> None:
        run_command(SCHEDULED_RETRAIN_COMMAND)
        self.next(self.monitor_after_retrain)

    @step
    def monitor_after_retrain(self) -> None:
        run_command(MONITOR_AFTER_RETRAIN_COMMAND)
        self.next(self.end)

    @step
    def end(self) -> None:
        return None


if __name__ == "__main__":
    ProductionRetrainingFlow()
