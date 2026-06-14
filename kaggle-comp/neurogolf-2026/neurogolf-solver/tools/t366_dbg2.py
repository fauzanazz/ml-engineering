import json
import numpy as np
from pathlib import Path
from collections import deque
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
e=ex[0]
gi=np.array(e['input']); go=np.array(e['output'])
H,W=gi.shape
A=gi[:H//2]; B=gi[H//2:]
print("A bg",bg(A),"B bg",bg(B))
# mk=B (bg0), tp=A (bg8)
mk,tp,mkbg,tpbg=B,A,bg(B),bg(A)
mkcolors=set(np.unique(mk[mk!=mkbg]).tolist())
print("mkcolors",mkcolors)
cs=comps(tp!=tpbg)
for cells in cs:
    rs=[r for r,_ in cells]; csl=[c for _,c in cells]
    mpix=[(r,c) for r,c in cells if tp[r,c] in mkcolors]
    if mpix:
        mc=tp[mpix[0][0],mpix[0][1]]
        mr=list(zip(*np.where(mk==mc)))
        print(f"shape bbox r{min(rs)}-{max(rs)} c{min(csl)}-{max(csl)} mc={mc} #mpix={len(mpix)} #mkpix={len(mr)}")
        print("  shape mpix", sorted(mpix))
        print("  mk pix   ", sorted(mr))
        sp=sorted(mpix); sm=sorted(mr)
        if len(sp)==len(sm):
            offs=set((b[0]-a[0],b[1]-a[1]) for a,b in zip(sp,sm))
            print("  offs",offs)
