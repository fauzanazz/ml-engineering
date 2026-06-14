import json, sys
from pathlib import Path
DATA=Path(__file__).resolve().parents[2]/"data"
def g(grid):
    return "\n".join("".join(str(c) for c in r) for r in grid)
t=int(sys.argv[1]); n=int(sys.argv[2]) if len(sys.argv)>2 else 3
d=json.loads((DATA/f"task{t:03d}.json").read_text())
print("train",len(d['train']),"test",len(d['test']),"arc-gen",len(d['arc-gen']))
for i,e in enumerate(d['train'][:n]):
    gi,go=e['input'],e['output']
    print(f"--- train{i} in {len(gi)}x{len(gi[0])} -> out {len(go)}x{len(go[0])} ---")
    print(g(gi)); print("=>"); print(g(go))
