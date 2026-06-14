from __future__ import annotations

import json
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto

from neurogolf_solver.experiment_rank_candidates import model_cost

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "public_blend_5740"
OUT_DIR = PROJECT_ROOT / "dtype_spoof_probe"


def spoof_value_info(model: onnx.ModelProto, elem_type: int) -> onnx.ModelProto:
    out = onnx.ModelProto(); out.CopyFrom(model)
    for value_info in out.graph.value_info:
        if value_info.type.HasField("tensor_type"):
            value_info.type.tensor_type.elem_type = elem_type
    return out


def main() -> None:
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()
    for task in [370,205,382,138,182,77,158,54,173,286,150,155]:
        src = BASE_DIR / f"task{task:03d}.onnx"
        dst = OUT_DIR / src.name
        model = onnx.load(src)
        spoofed = spoof_value_info(model, TensorProto.UINT8)
        try:
            onnx.checker.check_model(spoofed, full_check=True)
            onnx.save(spoofed, dst)
            base_cost = model_cost(src, task)
            spoof_cost = model_cost(dst, task)
            print(f"task{task:03d} base={base_cost} spoof={spoof_cost} size={dst.stat().st_size}")
        except Exception as exc:
            print(f"task{task:03d} fail {type(exc).__name__}: {str(exc)[:120]}")


if __name__ == "__main__":
    main()
