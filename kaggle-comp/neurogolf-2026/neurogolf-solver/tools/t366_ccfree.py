import json
import numpy as np
from pathlib import Path
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']+d['arc-gen']
def bg(g):
    v,c=np.unique(g,return_counts=True); return v[np.argmax(c)]
# Hypothesis: translate the ENTIRE template by t; valid t = every template marker-color pixel maps onto MK pixel of SAME color, and every MK marker pixel is covered. Then output = OR over valid t of (template non-bg shifted by t), on marker bg.
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
    tpmark=[(r,c,tp[r,c]) for r in range(R) for c in range(C) if tp[r,c] in mkcolors]
    tpbody=[(r,c,tp[r,c]) for r in range(R) for c in range(C) if tp[r,c]!=tpbg]
    out=np.full((R,C),mkbg,np.int64)
    mkn=(mk!=mkbg)
    covered=np.zeros((R,C),bool)
    for dr in range(-R,R+1):
        for dc in range(-C,C+1):
            if not tpmark: continue
            good=True
            for (r,c,col) in tpmark:
                rr,cc=r+dr,c+dc
                if not(0<=rr<R and 0<=cc<C) or mk[rr,cc]!=col:
                    good=False;break
            if not good: continue
            for (r,c,col) in tpbody:
                rr,cc=r+dr,c+dc
                if 0<=rr<R and 0<=cc<C:
                    out[rr,cc]=col
            for (r,c,col) in tpmark:
                covered[r+dr,c+dc]=True
    okk = out.shape==go.shape and np.array_equal(out,go)
    if okk: ok+=1
    else: ff.append(n)
print(f"CCFREE-ALLMARK PASS {ok}/{tot} fails(n={len(ff)}) {ff[:20]}")
