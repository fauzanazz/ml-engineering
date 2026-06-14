import json
import numpy as np
from pathlib import Path
from collections import deque,Counter
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']+d['arc-gen']
def bg(g):
    v,c=np.unique(g,return_counts=True); return v[np.argmax(c)]
def comps4(mask):
    seen=np.zeros_like(mask,bool); out=[]
    R,C=mask.shape
    for i in range(R):
        for j in range(C):
            if mask[i,j] and not seen[i,j]:
                q=deque([(i,j)]); seen[i,j]=True; cells=[]
                while q:
                    a,b=q.popleft(); cells.append((a,b))
                    for da,db in((1,0),(-1,0),(0,1),(0,-1)):
                        na,nb=a+da,b+db
                        if 0<=na<R and 0<=nb<C and mask[na,nb] and not seen[na,nb]:
                            seen[na,nb]=True; q.append((na,nb))
                out.append(cells)
    return out
stat=Counter()
nshapes=Counter(); nused=Counter(); nmarkcol=Counter(); markpx_per_shape=Counter()
ndecoy=Counter()
for e in ex:
    gi=np.array(e['input']);go=np.array(e['output'])
    if max(gi.shape)>30 or max(go.shape)>30: continue
    H,W=gi.shape
    if H>=W: A,B=gi[:H//2],gi[H//2:]
    else: A,B=gi[:,:W//2],gi[:,W//2:]
    sA=(A!=bg(A)).sum(); sB=(B!=bg(B)).sum()
    mk,tp=(A,B) if sA<=sB else (B,A)
    mkbg,tpbg=bg(mk),bg(tp)
    mkcolors=set(np.unique(mk[mk!=mkbg]).tolist())
    sh=comps4(tp!=tpbg)
    used=[c for c in sh if any(tp[r,cc] in mkcolors for r,cc in c)]
    decoy=len(sh)-len(used)
    nshapes[len(sh)]+=1; nused[len(used)]+=1; ndecoy[decoy]+=1
    nmarkcol[len(mkcolors)]+=1
    for c in used:
        mp=[(r,cc) for r,cc in c if tp[r,cc] in mkcolors]
        markpx_per_shape[len(mp)]+=1
print("total shapes/ex:",dict(sorted(nshapes.items())))
print("used shapes/ex:",dict(sorted(nused.items())))
print("decoy shapes/ex:",dict(sorted(ndecoy.items())))
print("marker colors/ex:",dict(sorted(nmarkcol.items())))
print("markerpx per used shape:",dict(sorted(markpx_per_shape.items())))
