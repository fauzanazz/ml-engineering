import json
import numpy as np
from pathlib import Path
from collections import Counter, deque
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']
def bg(g):
    v,c=np.unique(g,return_counts=True); return v[np.argmax(c)]
def comps(mask):
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
for n in range(3):
    e=ex[n]
    gi=np.array(e['input']); go=np.array(e['output'])
    H,W=gi.shape
    if go.shape==(H//2,W):
        A=gi[:H//2]; B=gi[H//2:]
    else:
        A=gi[:,:W//2]; B=gi[:,W//2:]
    aB,bB=bg(A),bg(B); obg=bg(go)
    if aB==obg: mk,tp,mkbg,tpbg=A,B,aB,bB
    else: mk,tp,mkbg,tpbg=B,A,bB,aB
    mkcolors=set(np.unique(mk[mk!=mkbg]).tolist())
    print(f"== ex{n} mkcolors {mkcolors} obg {obg} out {go.shape}")
    cs=comps(tp!=tpbg)
    for cells in cs:
        rs=[r for r,_ in cells]; csl=[c for _,c in cells]
        r0,r1,c0,c1=min(rs),max(rs),min(csl),max(csl)
        colset=Counter(tp[r,c] for r,c in cells)
        # marker pixel inside shape: color in mkcolors
        mpix=[(r,c,tp[r,c]) for r,c in cells if tp[r,c] in mkcolors]
        print(f"  shape bbox r{r0}-{r1} c{c0}-{c1} colors {dict(colset)} markerpix {mpix}")
