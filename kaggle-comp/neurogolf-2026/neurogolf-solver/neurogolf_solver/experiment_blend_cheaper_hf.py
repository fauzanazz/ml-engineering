from __future__ import annotations

import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from neurogolf_solver.experiment_rank_candidates import model_cost, task_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "public_blend_5740"
HF_DIR = PROJECT_ROOT / "full_public_solver"
OUT_DIR = PROJECT_ROOT / "experiment_full_public_cheaper_blend"


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()
    receipts = []
    for task_num in range(1, 401):
        name = f"task{task_num:03d}.onnx"
        base_path = BASE_DIR / name
        candidate_path = HF_DIR / name
        chosen = base_path
        if candidate_path.exists():
            base_cost = model_cost(base_path, task_num)
            candidate_cost = model_cost(candidate_path, task_num)
            if base_cost and candidate_cost and candidate_cost < base_cost:
                chosen = candidate_path
                receipts.append({
                    "task": task_num,
                    "base_cost": base_cost,
                    "candidate_cost": candidate_cost,
                    "delta_score": task_score(candidate_cost) - task_score(base_cost),
                })
        shutil.copy2(chosen, OUT_DIR / name)
    with ZipFile(PROJECT_ROOT / "submission.zip", "w", ZIP_DEFLATED) as archive:
        for model_path in sorted(OUT_DIR.glob("task*.onnx")):
            archive.write(model_path, model_path.name)
    print(json.dumps({"replacements": receipts, "count": len(receipts), "gain": sum(r["delta_score"] for r in receipts)}, indent=2))


if __name__ == "__main__":
    main()
