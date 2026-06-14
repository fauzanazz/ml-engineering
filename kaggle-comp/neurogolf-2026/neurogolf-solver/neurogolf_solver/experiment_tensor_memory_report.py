from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from neurogolf_solver.experiment_rank_candidates import examples


def encode(grid: list[list[int]]) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def tensor_memory_rows(model_path: Path, task_num: int) -> list[tuple[str, int, str]]:
    model = onnx.load(model_path)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    dtype = {}
    static = {}
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
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        dtype[item.name] = np_dtype
        static[item.name] = int(math.prod(dims)) * np.dtype(np_dtype).itemsize
    node_outputs = {node.name: list(node.output) for node in graph.node}
    opts = ort.SessionOptions(); opts.enable_profiling = True; opts.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    for ex in examples(task_num)[:3]:
        session.run(["output"], {"input": encode(ex["input"])})
    trace_path = session.end_profiling()
    trace = json.loads(Path(trace_path).read_text()); os.remove(trace_path)
    runtime = {}
    for event in trace:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        shapes = event["args"].get("output_type_shape")
        if not shapes:
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        outs = node_outputs.get(node_name, [])
        for i, shape_dict in enumerate(shapes):
            if i >= len(outs):
                continue
            name = outs[i]
            if name not in dtype:
                continue
            size = np.dtype(dtype[name]).itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            runtime[name] = max(runtime.get(name, 0), int(size))
    rows = []
    for name, static_size in static.items():
        if name in {"input", "output"}:
            continue
        rows.append((name, max(static_size, runtime.get(name, 0)), str(dtype[name])))
    return sorted(rows, key=lambda row: row[1], reverse=True)


def main() -> None:
    task_num = 370
    rows = tensor_memory_rows(Path("best_5743") / f"task{task_num:03d}.onnx", task_num)
    print("total", sum(row[1] for row in rows), "count", len(rows))
    for row in rows[:80]:
        print(row)


if __name__ == "__main__":
    main()
