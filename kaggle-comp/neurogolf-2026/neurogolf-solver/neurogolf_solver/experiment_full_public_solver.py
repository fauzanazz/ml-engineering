from __future__ import annotations

import json
import shutil
from pathlib import Path

import onnx

from neurogolf_solver import hf_enhanced_solver as solver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "data"
OUT_DIR = PROJECT_ROOT / "full_public_solver"


def full_public_task(task_num: int) -> dict:
    task = json.loads((DATA_DIR / f"task{task_num:03d}.json").read_text())
    examples = [example for example in task["train"] + task["test"] + task["arc-gen"] if max(len(example["input"]), len(example["input"][0])) <= 30]
    return {"train": examples, "test": [], "arc-gen": []}


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()
    results = []
    solver.ANALYTICAL_SOLVERS = solver.ANALYTICAL_SOLVERS
    for task_num in range(1, 401):
        task = full_public_task(task_num)
        ok, name, size = solver.solve_task(task_num, task, str(OUT_DIR), conv_budget=0.15)
        if ok:
            results.append((task_num, name, size))
            print(f"task{task_num:03d} {name} {size}")
    print(json.dumps({"count": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
