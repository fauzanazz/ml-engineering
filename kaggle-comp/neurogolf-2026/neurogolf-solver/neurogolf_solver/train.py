from __future__ import annotations

from neurogolf_solver.data import list_task_paths, summarize_task
from neurogolf_solver.submission import build_baseline_submission


def train_baseline() -> dict[str, int | str]:
    task_paths = list_task_paths()
    first_task_summary = summarize_task(1)
    submission = build_baseline_submission()
    return {
        "tasks": len(task_paths),
        "task001_train": first_task_summary["train"],
        "task001_test": first_task_summary["test"],
        "task001_arc_gen": first_task_summary["arc-gen"],
        "submission": str(submission),
    }


def main() -> None:
    print(train_baseline())


if __name__ == "__main__":
    main()
