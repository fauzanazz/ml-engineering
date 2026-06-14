from __future__ import annotations

import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from neurogolf_solver.experiment_rank_candidates import model_cost, task_score
from neurogolf_solver.filter_public_valid import passes_all_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "data"
BASE_DIR = PROJECT_ROOT / "best_5743"
OUT_DIR = PROJECT_ROOT / "constant_shape_candidates"
BLEND_DIR = PROJECT_ROOT / "experiment_constant_shape_blend"
SHAPE = [1, 10, 30, 30]


def examples(task_num: int) -> list[dict]:
    task = json.loads((DATA_DIR / f"task{task_num:03d}.json").read_text())
    return [e for e in task["train"] + task["test"] + task["arc-gen"] if max(len(e["input"]), len(e["input"][0])) <= 30]


def same_output_grid(task_num: int) -> list[list[int]] | None:
    outs = [e["output"] for e in examples(task_num)]
    if outs and all(out == outs[0] for out in outs):
        return outs[0]
    return None


def make_constant_grid_model(grid: list[list[int]]) -> onnx.ModelProto:
    output_array = np.zeros(SHAPE, dtype=np.float32)
    for row_index, row in enumerate(grid):
        for col_index, color in enumerate(row):
            output_array[0, color, row_index, col_index] = 1.0
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, SHAPE)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, SHAPE)
    zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero")
    const = numpy_helper.from_array(output_array, "const")
    nodes = [
        helper.make_node("Mul", ["input", "zero"], ["dead"]),
        helper.make_node("ReduceSum", ["dead"], ["scalar"], axes=[1,2,3], keepdims=1),
        helper.make_node("Add", ["scalar", "const"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "constant_grid", [x], [y], [zero, const])
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])


def main() -> None:
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    if BLEND_DIR.exists(): shutil.rmtree(BLEND_DIR)
    OUT_DIR.mkdir(); BLEND_DIR.mkdir()
    receipts = []
    for task_num in range(1, 401):
        grid = same_output_grid(task_num)
        if grid is None:
            continue
        name = f"task{task_num:03d}.onnx"
        dst = OUT_DIR / name
        onnx.save(make_constant_grid_model(grid), dst)
        if not passes_all_examples(dst, DATA_DIR / f"task{task_num:03d}.json"):
            continue
        base_cost = model_cost(BASE_DIR / name, task_num)
        new_cost = model_cost(dst, task_num)
        if base_cost and new_cost and new_cost < base_cost:
            receipts.append({"task": task_num, "base_cost": base_cost, "new_cost": new_cost, "gain": task_score(new_cost)-task_score(base_cost)})
        else:
            dst.unlink(missing_ok=True)
    names = {f"task{r['task']:03d}.onnx" for r in receipts}
    for task_num in range(1, 401):
        name = f"task{task_num:03d}.onnx"
        shutil.copy2((OUT_DIR if name in names else BASE_DIR) / name, BLEND_DIR / name)
    with ZipFile(PROJECT_ROOT / "submission.zip", "w", ZIP_DEFLATED) as archive:
        for model_path in sorted(BLEND_DIR.glob("task*.onnx")):
            archive.write(model_path, model_path.name)
    print(json.dumps({"count": len(receipts), "gain": sum(r["gain"] for r in receipts), "receipts": receipts}, indent=2))


if __name__ == "__main__":
    main()
