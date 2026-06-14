import json, sys
import numpy as np
from pathlib import Path
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
def show(g):
    for r in g: print("".join(str(c) for c in r))
ex=d['train']+d['test']
n=int(sys.argv[1]) if len(sys.argv)>1 else 0
e=ex[n]
gi=np.array(e['input']); go=np.array(e['output'])
print("IN",gi.shape,"OUT",go.shape)
print("--- input ---"); show(gi.tolist())
print("--- output ---"); show(go.tolist())
