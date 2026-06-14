import json, sys
import numpy as np
from pathlib import Path
from collections import deque
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']+d['arc-gen']
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

def solve(gi):
    H,W=gi.shape
    # try vertical then horizontal split, pick the one giving consistent solution
    cands=[]
    if H%2==0: cands.append(('V', gi[:H//2], gi[H//2:]))
    if W%2==0: cands.append(('H', gi[:,:W//2], gi[:,W//2:]))
    for axis,A,Bv in cands:
        aB,bB=bg(A),bg(Bv)
        for mk,tp,mkbg,tpbg in ((A,Bv,aB,bB),(Bv,A,bB,aB)):
            res=try_solve(mk,tp,mkbg,tpbg)
            if res is not None:
                return res
    return None

def try_solve(mk,tp,mkbg,tpbg):
    R,C=mk.shape
    mkmask=mk!=mkbg
    mkcolors=set(np.unique(mk[mkmask]).tolist())
    if not mkcolors: return None
    out=np.full((R,C),mkbg,dtype=gi_dtype)
    cs=comps(tp!=tpbg)
    used=False
    for cells in cs:
        mpix=[(r,c) for r,c in cells if tp[r,c] in mkcolors]
        if not mpix: 
            return None
        mc=tp[mpix[0][0],mpix[0][1]]
        if any(tp[r,c]!=mc for r,c in mpix): return None
        # marker region pixels of color mc
        mr=list(zip(*np.where(mk==mc)))
        if len(mr)!=len(mpix): return None
        # find translation: sort both by (r,c)
        sp=sorted(mpix); sm=sorted(mr)
        offs=set((b[0]-a[0], b[1]-a[1]) for a,b in zip(sp,sm))
        if len(offs)!=1: return None
        dr,dc=next(iter(offs))
        for r,c in cells:
            nr,nc=r+dr,c+dc
            if not(0<=nr<R and 0<=nc<C): return None
            out[nr,nc]=tp[r,c]
        used=True
    if not used: return None
    return out

global gi_dtype
gi_dtype=np.int64
ok=0; fails=[]
for n,e in enumerate(ex):
    gi=np.array(e['input']); go=np.array(e['output'])
    out=solve(gi)
    if out is not None and out.shape==go.shape and np.array_equal(out,go):
        ok+=1
    else:
        fails.append(n)
print(f"PASS {ok}/{len(ex)}  fails={fails[:20]} (total {len(fails)})")
