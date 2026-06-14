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

def try_solve(mk,tp,mkbg,tpbg):
    R,C=mk.shape
    mkcolors=set(np.unique(mk[mk!=mkbg]).tolist())
    if not mkcolors: return None
    shapes=[]
    for cells in comps4(tp!=tpbg):
        mcells=[(r,c) for r,c in cells if tp[r,c] in mkcolors]
        if not mcells: return None
        r0=min(r for r,_ in cells); c0=min(c for _,c in cells)
        relcells=[(r-r0,c-c0,int(tp[r,c])) for r,c in cells]
        relmark=[(r-r0,c-c0,int(tp[r,c])) for r,c in mcells]
        shapes.append((relcells,relmark))
    out=np.full((R,C),mkbg,dtype=np.int64)
    consumed=np.zeros((R,C),bool)
    nb_total=int((mk!=mkbg).sum())
    placed_marks=0
    for relcells,relmark in shapes:
        # slide: anchor on first mark
        for ar in range(-R,R+1):
            for ac in range(-C,C+1):
                # marker cells positions
                ok=True
                for (dr,dc,col) in relmark:
                    rr,cc=ar+dr,ac+dc
                    if not(0<=rr<R and 0<=cc<C) or mk[rr,cc]!=col:
                        ok=False; break
                if not ok: continue
                # also require: this is exactly a marker group (all surrounding handled by uniqueness). stamp shape
                for (dr,dc,col) in relcells:
                    rr,cc=ar+dr,ac+dc
                    if not(0<=rr<R and 0<=cc<C): 
                        ok=False; break
                if not ok: continue
                for (dr,dc,col) in relcells:
                    out[ar+dr,ac+dc]=col
                for (dr,dc,col) in relmark:
                    consumed[ar+dr,ac+dc]=True
                placed_marks+=len(relmark)
    # validity: all mk non-bg consumed exactly, none left
    if not np.array_equal(consumed, mk!=mkbg): return None
    return out

def solve(gi):
    H,W=gi.shape
    cands=[]
    if H%2==0: cands.append((gi[:H//2], gi[H//2:]))
    if W%2==0: cands.append((gi[:,:W//2], gi[:,W//2:]))
    for A,Bv in cands:
        aB,bB=bg(A),bg(Bv)
        for mk,tp,mkbg,tpbg in ((A,Bv,aB,bB),(Bv,A,bB,aB)):
            res=try_solve(mk,tp,mkbg,tpbg)
            if res is not None: return res
    return None

ok=0; fails=[]
for n,e in enumerate(ex):
    gi=np.array(e['input']); go=np.array(e['output'])
    out=solve(gi)
    if out is not None and out.shape==go.shape and np.array_equal(out,go):
        ok+=1
    else:
        fails.append(n)
print(f"PASS {ok}/{len(ex)}  fails={fails[:30]} (total {len(fails)})")
