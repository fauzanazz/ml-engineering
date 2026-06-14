from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "data"
TASK_COUNT = 400


def task_path(task_num: int, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    if task_num < 1 or task_num > TASK_COUNT:
        raise ValueError(f"task_num must be between 1 and {TASK_COUNT}: {task_num}")
    return data_dir / f"task{task_num:03d}.json"


def load_task(task_num: int, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, list[dict[str, Any]]]:
    path = task_path(task_num, data_dir)
    with path.open() as file:
        task = json.load(file)
    return task


def list_task_paths(data_dir: Path = DEFAULT_DATA_DIR) -> list[Path]:
    return sorted(data_dir.glob("task*.json"))


def summarize_task(task_num: int, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, int]:
    task = load_task(task_num, data_dir)
    return {split_name: len(examples) for split_name, examples in task.items()}
