"""Airflow retraining adapter for {{project_name}}.

Import-safe skeleton: if Airflow is not installed, placeholder objects preserve
the task order for tests and readers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

SCHEDULED_RETRAIN_COMMAND = """uv run python orchestration/retraining_job.py \
  --config configs/project.yaml \
  --set-active \
  --require-quality-gate \
  --output-path artifacts/reports/scheduled-retraining.json"""

MONITOR_AFTER_RETRAIN_COMMAND = """uv run python orchestration/monitoring_job.py \
  --input-path artifacts/reports/scheduled-retraining.json \
  --output-path artifacts/reports/monitoring.json"""

VALIDATE_CONFIG_COMMAND = "test -f configs/project.yaml"

try:
    from airflow import DAG  # type: ignore[import-not-found]
    from airflow.operators.bash import BashOperator  # type: ignore[import-not-found]
except ImportError:

    class PlaceholderTask:
        def __init__(self, task_id: str, bash_command: str) -> None:
            self.task_id = task_id
            self.bash_command = bash_command

        def __rshift__(self, other: "PlaceholderTask") -> "PlaceholderTask":
            return other

    class PlaceholderDAG:
        def __init__(self, dag_id: str, schedule: str, start_date: datetime, catchup: bool, tags: list[str]) -> None:
            self.dag_id = dag_id
            self.schedule = schedule
            self.start_date = start_date
            self.catchup = catchup
            self.tags = tags
            self.tasks: list[PlaceholderTask] = []

        def add_task(self, task: PlaceholderTask) -> None:
            self.tasks.append(task)

    def DAG(  # type: ignore[no-redef]
        dag_id: str,
        schedule: str,
        start_date: datetime,
        catchup: bool,
        tags: list[str],
        **_: Any,
    ) -> PlaceholderDAG:
        return PlaceholderDAG(dag_id, schedule, start_date, catchup, tags)

    class BashOperator(PlaceholderTask):  # type: ignore[no-redef]
        def __init__(self, task_id: str, bash_command: str, dag: PlaceholderDAG, **_: Any) -> None:
            super().__init__(task_id=task_id, bash_command=bash_command)
            dag.add_task(self)


dag = DAG(
    dag_id="{{package_name}}_retraining",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "retraining", "{{backend}}"],
)

validate_config = BashOperator(task_id="validate_config", bash_command=VALIDATE_CONFIG_COMMAND, dag=dag)
scheduled_retrain = BashOperator(task_id="scheduled_retrain", bash_command=SCHEDULED_RETRAIN_COMMAND, dag=dag)
monitor_after_retrain = BashOperator(task_id="monitor_after_retrain", bash_command=MONITOR_AFTER_RETRAIN_COMMAND, dag=dag)

validate_config >> scheduled_retrain >> monitor_after_retrain
