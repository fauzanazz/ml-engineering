from __future__ import annotations

import json
import shutil
from pathlib import Path

import onnx

from neurogolf_solver import hf_enhanced_solver as solver
from neurogolf_solver.experiment_rank_candidates import model_cost, task_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "data"
BASE_DIR = PROJECT_ROOT / "public_blend_5740"
OUT_DIR = PROJECT_ROOT / "targeted_full_public"
TASKS = [370,205,382,138,182,77,158,54,173,286,396,364,80,285,92,133,19,349,51,264,280,161,64,383,71,387,202,110,284,5,324,13,18,358,85,290,187,34,204,216]


def filtered_task(task_num: int) -> dict:
    task = json.loads((DATA_DIR / f"task{task_num:03d}.json").read_text())
    examples = [example for example in task["train"] + task["test"] + task["arc-gen"] if max(len(example["input"]), len(example["input"][0])) <= 30]
    return {"train": examples, "test": [], "arc-gen": []}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    receipts = []
    for task_num in TASKS:
        path = OUT_DIR / f"task{task_num:03d}.onnx"
        if path.exists():
            continue
        ok, solver_name, size = solver.solve_task(task_num, filtered_task(task_num), str(OUT_DIR), conv_budget=2.0)
        if not ok:
            print(f"task{task_num:03d} no_fit")
            continue
        base_cost = model_cost(BASE_DIR / path.name, task_num)
        new_cost = model_cost(path, task_num)
        gain = task_score(new_cost) - task_score(base_cost) if base_cost and new_cost else None
        row = {"task": task_num, "solver": solver_name, "size": size, "base_cost": base_cost, "new_cost": new_cost, "gain": gain}
        receipts.append(row)
        print(json.dumps(row))
    print("SUMMARY", json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
