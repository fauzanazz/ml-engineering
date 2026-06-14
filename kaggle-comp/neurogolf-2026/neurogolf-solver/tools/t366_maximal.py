import json
import numpy as np
from pathlib import Path
from collections import deque
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
ok=0;tot=0;ff=[]
for n,e in enumerate(ex):
    gi=np.array(e['input']);go=np.array(e['output'])
    if max(gi.shape)>30 or max(go.shape)>30: continue
    tot+=1
    H,W=gi.shape
    if H>=W: A,B=gi[:H//2],gi[H//2:]
    else: A,B=gi[:,:W//2],gi[:,W//2:]
    sA=(A!=bg(A)).sum(); sB=(B!=bg(B)).sum()
    mk,tp=(A,B) if sA<=sB else (B,A)
    mkbg,tpbg=bg(mk),bg(tp)
    R,C=mk.shape
    mkcolors=set(np.unique(mk[mk!=mkbg]).tolist())
    shapes=[]
    for cells in comps4(tp!=tpbg):
        mcells=[(r,c) for r,c in cells if tp[r,c] in mkcolors]
        if not mcells: continue
        r0=min(r for r,_ in cells); c0=min(c for _,c in cells)
        relcells=[(r-r0,c-c0,int(tp[r,c])) for r,c in cells]
        relmark=[(r-r0,c-c0,int(tp[r,c])) for r,c in mcells]
        shapes.append((relcells,relmark))
    # enumerate all valid placements (shape idx, dr,dc, set of mk cells)
    placements=[]
    for si,(relcells,relmark) in enumerate(shapes):
        for ar in range(-R,R+1):
            for ac in range(-C,C+1):
                good=True; mset=set()
                for (dr,dc,col) in relmark:
                    rr,cc=ar+dr,ac+dc
                    if not(0<=rr<R and 0<=cc<C) or mk[rr,cc]!=col: good=False;break
                    mset.add((rr,cc))
                if not good: continue
                for (dr,dc,col) in relcells:
                    rr,cc=ar+dr,ac+dc
                    if not(0<=rr<R and 0<=cc<C): good=False;break
                if not good: continue
                placements.append((si,ar,ac,frozenset(mset),len(relmark)))
    # keep maximal: drop placement whose mk-set is subset of another placement's mk-set (strict)
    keep=[]
    for p in placements:
        sub=False
        for q in placements:
            if p is q: continue
            if p[3] < q[3]:
                sub=True;break
        if not sub: keep.append(p)
    out=np.full((R,C),mkbg,np.int64)
    for si,ar,ac,mset,_ in keep:
        relcells,relmark=shapes[si]
        for (dr,dc,col) in relcells:
            out[ar+dr,ac+dc]=col
    okk=out.shape==go.shape and np.array_equal(out,go)
    if okk: ok+=1
    else: ff.append(n)
print(f"MAXIMAL {ok}/{tot} nfail={len(ff)} {ff[:20]}")
