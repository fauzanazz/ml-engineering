"""Verify Rust ResT forward matches PyTorch teacher.

Reads JSONL from `rest_probe` (each line: {value, top, f}), runs the Python
teacher on the identical feature vector `f`, and compares value + top policy.

Usage: rest_probe <w> | uv run parity_check.py <teacher.safetensors>
"""

import json
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from model import WallNetResT

teacher_path = sys.argv[1]
m = WallNetResT(channels=64, blocks="RRTRRT")
m.load_state_dict(load_file(teacher_path))
m.eval()

max_vdiff = 0.0
rows = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    f = torch.tensor(rec["f"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, v = m(f)
        v = float(v.item())
        probs = F.softmax(logits, dim=-1)[0]
    py_top = torch.topk(probs, 3)
    rust_v = rec["value"]
    vdiff = abs(rust_v - v)
    max_vdiff = max(max_vdiff, vdiff)
    rows += 1
    # py top indices are me-frame; rust top are abs-frame — compare value precisely,
    # and report py top prob magnitude as a sanity signal.
    print(
        f"row {rows}: rust_v={rust_v:+.5f}  py_v={v:+.5f}  |dv|={vdiff:.2e}   "
        f"py_top_prob={py_top.values[0]:.4f}  rust_top={rec['top'][0]}"
    )

print(f"\nMAX |value diff| over {rows} positions: {max_vdiff:.2e}")
print("PARITY OK" if max_vdiff < 1e-3 else "PARITY FAIL (>1e-3)")
