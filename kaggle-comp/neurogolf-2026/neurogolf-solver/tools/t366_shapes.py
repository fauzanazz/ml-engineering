import json
import numpy as np
from pathlib import Path
from collections import Counter
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']+d['arc-gen']
ish=Counter(); osh=Counter(); axc=Counter()
for e in ex:
    gi=np.array(e['input']);go=np.array(e['output'])
    H,W=gi.shape;OH,OW=go.shape
    ish[(H,W)]+=1; osh[(OH,OW)]+=1
    ax='V' if (OH==H//2 and OW==W) else 'H'
    axc[ax]+=1
print("input shapes:",dict(sorted(ish.items())))
print("output shapes:",dict(sorted(osh.items())))
print("axis:",dict(axc))
print("max H",max(h for h,w in ish),"max W",max(w for h,w in ish))
