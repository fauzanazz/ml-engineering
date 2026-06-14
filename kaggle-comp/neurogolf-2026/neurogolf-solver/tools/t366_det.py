import json
import numpy as np
from pathlib import Path
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']+d['arc-gen']
def bg(g):
    v,c=np.unique(g,return_counts=True); return v[np.argmax(c)]
rows=[]
for e in ex:
    gi=np.array(e['input']);go=np.array(e['output'])
    H,W=gi.shape;OH,OW=go.shape
    if max(gi.shape)>30 or max(go.shape)>30: continue
    ax='V' if (OH==H//2 and OW==W) else 'H'
    # heuristic: split along the LONGER axis? compare H vs W
    longer = 'V' if H>=W else 'H'   # V means split rows (top/bottom) when H is bigger
    rows.append((ax, longer, H, W))
from collections import Counter
print("ax vs longer:",Counter((a,l) for a,l,_,_ in rows))
# does split always halve the longer dimension?
print("H>W when V:",Counter(('V' if H>=W else 'H') for a,l,H,W in rows if a=='V'))
print("H<W when H:",Counter(('V' if H>=W else 'H') for a,l,H,W in rows if a=='H'))
