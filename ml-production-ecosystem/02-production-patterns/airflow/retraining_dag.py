"""Airflow-style DAG skeleton for production retraining orchestration.

This file is import-safe without Airflow installed. Placeholder objects keep task
order and command examples readable until real Airflow runtime is added.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

SCHEDULED_RETRAIN_COMMAND = """uv run production-scheduled-retrain \
  --config configs/foundation-recommender.yaml \
  --set-active \
  --require-quality-gate \
  --output-path 02-production-patterns/reports/scheduled-retraining.json"""

MONITOR_AFTER_RETRAIN_COMMAND = """uv run production-monitor \
  --base-url http://127.0.0.1:8000 \
  --max-error-count 0 \
  --max-drift-score 0.2 \
  --max-latency-ms-last 100"""

VALIDATE_CONFIG_COMMAND = "test -f configs/foundation-recommender.yaml"

try:
    from airflow import DAG  # type: ignore[import-not-found]
    from airflow.operators.bash import BashOperator  # type: ignore[import-not-found]
except ImportError:

    class PlaceholderTask:
        def __init__(self, task_id: str, bash_command: str) -> None:
            self.task_id = task_id
            self.bash_command = bash_command
            self.downstream_task_ids: list[str] = []

        def __rshift__(self, other: "PlaceholderTask") -> "PlaceholderTask":
            self.downstream_task_ids.append(other.task_id)
            return other

    class PlaceholderDAG:
        def __init__(self, dag_id: str, schedule: str, start_date: datetime, catchup: bool, tags: list[str]) -> None:
            self.dag_id = dag_id
            self.schedule = schedule
            self.start_date = start_date
            self.catchup = catchup
            self.tags = tags
            self.tasks: list[PlaceholderTask] = []

        def add_task(self, task: PlaceholderTask) -> PlaceholderTask:
            self.tasks.append(task)
            return task

    def DAG(  # type: ignore[no-redef]
        dag_id: str,
        schedule: str,
        start_date: datetime,
        catchup: bool,
        tags: list[str],
        **_: Any,
    ) -> PlaceholderDAG:
        return PlaceholderDAG(
            dag_id=dag_id,
            schedule=schedule,
            start_date=start_date,
            catchup=catchup,
            tags=tags,
        )

    class BashOperator(PlaceholderTask):  # type: ignore[no-redef]
        def __init__(self, task_id: str, bash_command: str, dag: PlaceholderDAG, **_: Any) -> None:
            super().__init__(task_id=task_id, bash_command=bash_command)
            dag.add_task(self)


dag = DAG(
    dag_id="production_retraining",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml-production", "retraining"],
)

validate_config = BashOperator(
    task_id="validate_config",
    bash_command=VALIDATE_CONFIG_COMMAND,
    dag=dag,
)

scheduled_retrain = BashOperator(
    task_id="scheduled_retrain",
    bash_command=SCHEDULED_RETRAIN_COMMAND,
    dag=dag,
)

monitor_after_retrain = BashOperator(
    task_id="monitor_after_retrain",
    bash_command=MONITOR_AFTER_RETRAIN_COMMAND,
    dag=dag,
)

# Task order: validate_config >> scheduled_retrain >> monitor_after_retrain
validate_config >> scheduled_retrain >> monitor_after_retrain
