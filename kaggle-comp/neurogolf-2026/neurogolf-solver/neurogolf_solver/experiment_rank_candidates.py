from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "data"
BASE_DIR = PROJECT_ROOT / "best_5743"
FUSED_DIR = PROJECT_ROOT / "public_open_5689"


def task_score(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def examples(task_num: int) -> list[dict]:
    task = json.loads((DATA_DIR / f"task{task_num:03d}.json").read_text())
    return task["train"] + task["test"] + task["arc-gen"]


def calculate_params(model: onnx.ModelProto) -> int:
    total = 0
    for init in model.graph.initializer:
        total += math.prod(init.dims) if init.dims else 1
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                total += math.prod(attr.t.dims) if attr.t.dims else 1
            elif attr.name == "value_floats":
                total += len(attr.floats)
            elif attr.name == "value_ints":
                total += len(attr.ints)
    return int(total)


def calculate_memory(model_path: Path, task_examples: list[dict], runs: int = 3) -> int:
    model = onnx.load(model_path)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    tensor_dtype = {}
    tensor_static = {}
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        ok = True
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                ok = False
                break
            dims.append(dim.dim_value)
        if not ok:
            continue
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_dtype[item.name] = dtype
        tensor_static[item.name] = math.prod(dims) * np.dtype(dtype).itemsize

    node_outputs = {node.name: list(node.output) for node in graph.node}
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    for example in task_examples[:runs]:
        array = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for row_index, row in enumerate(example["input"]):
            for col_index, color in enumerate(row):
                array[0, color, row_index, col_index] = 1.0
        session.run(["output"], {"input": array})
    trace_path = session.end_profiling()
    trace = json.loads(Path(trace_path).read_text())
    os.remove(trace_path)
    runtime = {}
    for event in trace:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        shapes = event["args"].get("output_type_shape")
        if not shapes:
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        for index, shape_dict in enumerate(shapes):
            outs = node_outputs.get(node_name, [])
            if index >= len(outs):
                continue
            name = outs[index]
            if name not in tensor_dtype:
                continue
            dtype = tensor_dtype[name]
            value = np.dtype(dtype).itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            runtime[name] = max(runtime.get(name, 0), value)
    return int(sum(max(static, runtime.get(name, 0)) for name, static in tensor_static.items() if name not in {"input", "output"}))


def model_cost(model_path: Path, task_num: int) -> int | None:
    try:
        model = onnx.load(model_path)
        return calculate_params(model) + calculate_memory(model_path, examples(task_num))
    except Exception:
        return None


def main() -> None:
    rows = []
    for task_num in range(1, 401):
        path = BASE_DIR / f"task{task_num:03d}.onnx"
        cost = model_cost(path, task_num)
        if cost is None:
            continue
        rows.append((task_num, cost, task_score(cost), path.stat().st_size))
    rows.sort(key=lambda row: row[1], reverse=True)
    print("Top expensive tasks")
    for task_num, cost, score, size in rows[:40]:
        print(f"task{task_num:03d} cost={cost:>10} score={score:6.2f} size={size:>8}")
    print("predicted", sum(row[2] for row in rows), "tasks", len(rows))


if __name__ == "__main__":
    main()
