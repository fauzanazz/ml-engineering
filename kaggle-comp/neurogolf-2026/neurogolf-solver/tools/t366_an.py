import json
import numpy as np
from pathlib import Path
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']
for n in range(3):
    e=ex[n]
    gi=np.array(e['input']); go=np.array(e['output'])
    H,W=gi.shape; OH,OW=go.shape
    vertical = (OH==H//2 and OW==W)  # halves stacked
    horizontal = (OW==W//2 and OH==H)
    print(f"== ex{n} IN {gi.shape} OUT {go.shape} vert={vertical} horiz={horizontal}")
    if vertical:
        A=gi[:H//2]; B=gi[H//2:]
    elif horizontal:
        A=gi[:,:W//2]; B=gi[:,W//2:]
    else:
        print("  neither"); continue
    # which half equals output bg-wise: find background = most common
    def bg(g): v,c=np.unique(g,return_counts=True); return v[np.argmax(c)]
    print("  A bg",bg(A),"B bg",bg(B),"O bg",bg(go),"colors A",np.unique(A),"B",np.unique(B),"O",np.unique(go))
    # is output == B with overlays?
    print("  B==O frac", (B==go).mean())
