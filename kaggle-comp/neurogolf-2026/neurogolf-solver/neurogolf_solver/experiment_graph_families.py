from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import onnx

from neurogolf_solver.experiment_rank_candidates import model_cost, task_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "best_5743"


def signature(path: Path) -> str:
    model = onnx.load(path)
    counts = Counter(node.op_type for node in model.graph.node)
    return ",".join(f"{op}:{counts[op]}" for op in sorted(counts))


def main() -> None:
    families = defaultdict(list)
    for task_num in range(1, 401):
        path = BASE_DIR / f"task{task_num:03d}.onnx"
        cost = model_cost(path, task_num)
        if not cost:
            continue
        families[signature(path)].append((task_num, cost, task_score(cost), path.stat().st_size))
    rows = []
    for sig, items in families.items():
        rows.append({
            "count": len(items),
            "total_cost": sum(item[1] for item in items),
            "avg_score": sum(item[2] for item in items) / len(items),
            "tasks": [item[0] for item in sorted(items, key=lambda x: x[1], reverse=True)[:20]],
            "signature": sig,
        })
    rows.sort(key=lambda row: row["total_cost"], reverse=True)
    for row in rows[:25]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
