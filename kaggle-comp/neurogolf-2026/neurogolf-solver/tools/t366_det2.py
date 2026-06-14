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
def stamp(mk,tp):
    R,C=mk.shape
    mkbg=bg(mk); tpbg=bg(tp)
    mkcolors=set(np.unique(mk[mk!=mkbg]).tolist())
    shapes=[]
    for cells in comps4(tp!=tpbg):
        mcells=[(r,c) for r,c in cells if tp[r,c] in mkcolors]
        if not mcells: continue
        r0=min(r for r,_ in cells); c0=min(c for _,c in cells)
        relcells=[(r-r0,c-c0,int(tp[r,c])) for r,c in cells]
        relmark=[(r-r0,c-c0,int(tp[r,c])) for r,c in mcells]
        shapes.append((relcells,relmark,len(mcells)))
    shapes.sort(key=lambda s:-s[2])
    out=np.full((R,C),mkbg,dtype=np.int64)
    consumed=np.zeros((R,C),bool)
    for relcells,relmark,ncnt in shapes:
        for ar in range(-R,R+1):
            for ac in range(-C,C+1):
                ok=True
                for (dr,dc,col) in relmark:
                    rr,cc=ar+dr,ac+dc
                    if not(0<=rr<R and 0<=cc<C) or mk[rr,cc]!=col or consumed[rr,cc]:
                        ok=False;break
                if not ok: continue
                for (dr,dc,col) in relcells:
                    rr,cc=ar+dr,ac+dc
                    if not(0<=rr<R and 0<=cc<C): ok=False;break
                if not ok: continue
                for (dr,dc,col) in relcells: out[ar+dr,ac+dc]=col
                for (dr,dc,col) in relmark: consumed[ar+dr,ac+dc]=True
    return out, np.array_equal(consumed, mk!=mkbg)

def solve(gi):
    H,W=gi.shape
    if H>=W:
        A,B=gi[:H//2],gi[H//2:]   # vertical
    else:
        A,B=gi[:,:W//2],gi[:,W//2:]
    aA,aB=A,B
    # marker = sparser half
    sA=(A!=bg(A)).sum(); sB=(B!=bg(B)).sum()
    if sA<=sB: mk,tp=A,B
    else: mk,tp=B,A
    out,full=stamp(mk,tp)
    return out

ok=0;tot=0;ff=[]
for n,e in enumerate(ex):
    gi=np.array(e['input']);go=np.array(e['output'])
    if max(gi.shape)>30 or max(go.shape)>30: continue
    tot+=1
    out=solve(gi)
    if out.shape==go.shape and np.array_equal(out,go): ok+=1
    else: ff.append(n)
print(f"DET PASS {ok}/{tot} fails={ff[:30]} (n={len(ff)})")
