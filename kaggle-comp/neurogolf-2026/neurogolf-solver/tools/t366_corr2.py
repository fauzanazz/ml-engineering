import json,sys
import numpy as np
from pathlib import Path
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']+d['arc-gen']
def bg(g):
    v,c=np.unique(g,return_counts=True); return v[np.argmax(c)]
for n in [9,10,11,12,13,16]:
    e=ex[n]
    gi=np.array(e['input']);go=np.array(e['output'])
    if max(gi.shape)>30 or max(go.shape)>30: 
        print(n,"skip"); continue
    H,W=gi.shape
    print(f"=== ex{n} IN {gi.shape} OUT {go.shape}")
    for r in gi: print(" I","".join(map(str,r)))
    for r in go: print(" O","".join(map(str,r)))
