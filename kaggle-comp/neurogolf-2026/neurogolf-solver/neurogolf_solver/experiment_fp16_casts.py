from __future__ import annotations

import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import onnx
from onnx import TensorProto, helper

from neurogolf_solver.experiment_rank_candidates import model_cost, task_score
from neurogolf_solver.filter_public_valid import passes_all_examples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "best_5743"
DATA_DIR = PROJECT_ROOT.parent / "data"
OUT_DIR = PROJECT_ROOT / "fp16_cast_candidates"
BLEND_DIR = PROJECT_ROOT / "experiment_fp16_cast_blend"


def narrow_casts(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    out = onnx.ModelProto(); out.CopyFrom(model)
    changed = 0
    for node in out.graph.node:
        if node.op_type != "Cast":
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if attrs.get("to") != TensorProto.FLOAT:
            continue
        del node.attribute[:]
        node.attribute.extend([helper.make_attribute("to", TensorProto.FLOAT16)])
        changed += 1
    cast_outputs = {node.output[0] for node in out.graph.node if node.op_type == "Cast" and node.output}
    for value_info in out.graph.value_info:
        if value_info.name in cast_outputs and value_info.type.HasField("tensor_type") and value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
    return out, changed


def main() -> None:
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    if BLEND_DIR.exists(): shutil.rmtree(BLEND_DIR)
    OUT_DIR.mkdir(); BLEND_DIR.mkdir()
    receipts = []
    for task_num in range(1, 401):
        name = f"task{task_num:03d}.onnx"
        src = BASE_DIR / name
        try:
            model, changed = narrow_casts(onnx.load(src))
            if not changed:
                continue
            dst = OUT_DIR / name
            onnx.checker.check_model(model, full_check=True)
            onnx.save(model, dst)
            if not passes_all_examples(dst, DATA_DIR / f"task{task_num:03d}.json"):
                continue
            base_cost = model_cost(src, task_num)
            new_cost = model_cost(dst, task_num)
            if base_cost and new_cost and new_cost < base_cost:
                receipts.append({"task": task_num, "changed": changed, "base_cost": base_cost, "new_cost": new_cost, "gain": task_score(new_cost)-task_score(base_cost)})
            else:
                dst.unlink(missing_ok=True)
        except Exception:
            continue
    names = {f"task{r['task']:03d}.onnx" for r in receipts}
    for task_num in range(1, 401):
        name = f"task{task_num:03d}.onnx"
        shutil.copy2((OUT_DIR if name in names else BASE_DIR) / name, BLEND_DIR / name)
    with ZipFile(PROJECT_ROOT / "submission.zip", "w", ZIP_DEFLATED) as archive:
        for model_path in sorted(BLEND_DIR.glob("task*.onnx")):
            archive.write(model_path, model_path.name)
    print(json.dumps({"count": len(receipts), "gain": sum(r["gain"] for r in receipts), "receipts": receipts[:80]}, indent=2))


if __name__ == "__main__":
    main()
