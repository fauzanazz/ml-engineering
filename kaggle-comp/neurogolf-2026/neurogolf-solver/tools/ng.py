"""Authoritative per-task cost/verify using official neurogolf_utils."""
from __future__ import annotations
import sys, types, os, json, math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATA_DIR = REPO_ROOT / "data"

def _stub(name, attrs=None):
    m = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

# stub heavy/irrelevant imports
_ip = _stub("IPython")
_ipd = _stub("IPython.display", {"display": lambda *a, **k: None, "FileLink": lambda *a, **k: None})
_ip.display = _ipd
_mpl = _stub("matplotlib")
_stub("matplotlib.pyplot", {"figure": lambda *a, **k: None})
_mpl.pyplot = sys.modules["matplotlib.pyplot"]
_stub("onnx_tool", {"model_profile": lambda *a, **k: None})

sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))
import neurogolf_utils as ng  # noqa: E402
ng._NEUROGOLF_DIR = str(DATA_DIR.resolve()) + "/"

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

def examples(task_num: int):
    return ng.load_examples(task_num)

def verify_and_cost(onnx_path: str, task_num: int, profile_runs: int = 8):
    """Return (all_pass, n_pass, n_total, cost, params, memory) using official scoring."""
    ex = ng.load_examples(task_num)
    model = onnx.load(onnx_path)
    sanitized = ng.sanitize_model(onnx.load(onnx_path))
    if sanitized is None:
        return (False, 0, 0, None, None, None)
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3
    pfx = f"tmpprof_{task_num:03d}_"
    opts.profile_file_prefix = pfx
    sess = ort.InferenceSession(sanitized.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    allex = ex["train"] + ex["test"] + ex["arc-gen"]
    n_pass = n_total = 0
    used = 0
    for e in allex:
        b = ng.convert_to_numpy(e)
        if not b:
            continue
        n_total += 1
        try:
            out = sess.run(["output"], {"input": b["input"]})[0]
            ok = np.array_equal((out > 0.0).astype(float), b["output"])
        except Exception:
            ok = False
        n_pass += int(ok)
        used += 1
    trace = sess.end_profiling()
    try:
        mem, params = ng.score_network(sanitized, trace)
    finally:
        try: os.remove(trace)
        except OSError: pass
    if mem is None or params is None:
        cost = None
    else:
        cost = mem + params
    return (n_pass == n_total and n_total > 0, n_pass, n_total, cost, params, mem)

def task_points(cost):
    if cost is None:
        return 0.0
    return max(1.0, 25.0 - math.log(max(1.0, cost)))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--tasks", default="1-400")
    p.add_argument("--out", default="")
    a = p.parse_args()
    lo, hi = (a.tasks.split("-") + [a.tasks])[:2]
    rng = range(int(lo), int(hi) + 1)
    rows = []
    total = 0.0
    for t in rng:
        path = Path(a.dir) / f"task{t:03d}.onnx"
        if not path.exists():
            continue
        ap, npas, ntot, cost, params, mem = verify_and_cost(str(path), t)
        pts = task_points(cost) if ap else 0.0
        total += pts
        rows.append({"task": t, "pass": ap, "n": f"{npas}/{ntot}", "cost": cost,
                     "params": params, "mem": mem, "pts": round(pts, 3)})
        print(f"task{t:03d} pass={ap} {npas}/{ntot} cost={cost} pts={pts:.3f}", flush=True)
    rows.sort(key=lambda r: (r["cost"] or 0), reverse=True)
    print("TOTAL", round(total, 3))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
