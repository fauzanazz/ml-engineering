import json,sys
import numpy as np
from pathlib import Path
from collections import deque
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']+d['arc-gen']
def bg(g):
    v,c=np.unique(g,return_counts=True); return v[np.argmax(c)]
def comps(mask,conn8=True):
    seen=np.zeros_like(mask,bool); out=[]
    R,C=mask.shape
    nb=((1,0),(-1,0),(0,1),(0,-1))+(((1,1),(1,-1),(-1,1),(-1,-1)) if conn8 else ())
    for i in range(R):
        for j in range(C):
            if mask[i,j] and not seen[i,j]:
                q=deque([(i,j)]); seen[i,j]=True; cells=[]
                while q:
                    a,b=q.popleft(); cells.append((a,b))
                    for da,db in nb:
                        na,nbb=a+da,b+db
                        if 0<=na<R and 0<=nbb<C and mask[na,nbb] and not seen[na,nbb]:
                            seen[na,nbb]=True; q.append((na,nbb))
                out.append(cells)
    return out
n=int(sys.argv[1])
e=ex[n]
gi=np.array(e['input']); go=np.array(e['output'])
H,W=gi.shape
print("IN",gi.shape,"OUT",go.shape)
splits=[]
if H%2==0: splits.append(("V",gi[:H//2],gi[H//2:]))
if W%2==0: splits.append(("H",gi[:,:W//2],gi[:,W//2:]))
for ax,A,B in splits:
    print("axis",ax,"A bg",bg(A),"B bg",bg(B),"obg",bg(go))
e0=ex[n]
def dump(name,g):
    print(name)
    for r in g: print(" ","".join(map(str,r)))
dump("IN",gi); dump("OUT",go)
