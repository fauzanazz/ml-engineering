from __future__ import annotations

import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import onnx

from neurogolf_solver import hf_enhanced_solver as solver
from neurogolf_solver.experiment_rank_candidates import model_cost, task_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "data"
BASE_DIR = PROJECT_ROOT / "public_blend_5740"
OUT_DIR = PROJECT_ROOT / "analytical_public"
BLEND_DIR = PROJECT_ROOT / "experiment_analytical_blend"


def filtered_examples(task_num: int) -> list[dict]:
    task = json.loads((DATA_DIR / f"task{task_num:03d}.json").read_text())
    return [example for example in task["train"] + task["test"] + task["arc-gen"] if max(len(example["input"]), len(example["input"][0])) <= 30]


def task_data(task_num: int) -> dict:
    return {"train": filtered_examples(task_num), "test": []}


def main() -> None:
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    if BLEND_DIR.exists(): shutil.rmtree(BLEND_DIR)
    OUT_DIR.mkdir(); BLEND_DIR.mkdir()
    receipts = []
    for task_num in range(1, 401):
        td = task_data(task_num)
        for solver_name, solver_fn in solver.ANALYTICAL_SOLVERS:
            model = None
            try:
                model = solver_fn(td)
            except Exception:
                pass
            if model is None:
                continue
            candidate_path = OUT_DIR / f"task{task_num:03d}.onnx"
            onnx.save(model, candidate_path)
            if not solver.validate(str(candidate_path), td):
                candidate_path.unlink(missing_ok=True)
                continue
            base_path = BASE_DIR / candidate_path.name
            base_cost = model_cost(base_path, task_num)
            candidate_cost = model_cost(candidate_path, task_num)
            if base_cost and candidate_cost and candidate_cost < base_cost:
                receipts.append({"task": task_num, "solver": solver_name, "base_cost": base_cost, "candidate_cost": candidate_cost, "gain": task_score(candidate_cost)-task_score(base_cost)})
            else:
                candidate_path.unlink(missing_ok=True)
            break
    replacements = {f"task{r['task']:03d}.onnx" for r in receipts}
    for task_num in range(1, 401):
        name = f"task{task_num:03d}.onnx"
        src = OUT_DIR / name if name in replacements else BASE_DIR / name
        shutil.copy2(src, BLEND_DIR / name)
    with ZipFile(PROJECT_ROOT / "submission.zip", "w", ZIP_DEFLATED) as archive:
        for model_path in sorted(BLEND_DIR.glob("task*.onnx")):
            archive.write(model_path, model_path.name)
    print(json.dumps({"count": len(receipts), "gain": sum(r["gain"] for r in receipts), "receipts": receipts}, indent=2))


if __name__ == "__main__":
    main()
