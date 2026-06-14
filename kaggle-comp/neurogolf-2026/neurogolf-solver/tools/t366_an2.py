import json
import numpy as np
from pathlib import Path
from collections import Counter
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']
def bg(g):
    v,c=np.unique(g,return_counts=True); return v[np.argmax(c)]
for n in range(3):
    e=ex[n]
    gi=np.array(e['input']); go=np.array(e['output'])
    H,W=gi.shape
    if go.shape==(H//2,W):
        A=gi[:H//2]; B=gi[H//2:]; axis='V'
    else:
        A=gi[:,:W//2]; B=gi[:,W//2:]; axis='H'
    aB,bB=bg(A),bg(B)
    obg=bg(go)
    # marker half = the one whose bg == output bg
    if aB==obg: mk,tp,mkbg,tpbg=A,B,aB,bB
    else: mk,tp,mkbg,tpbg=B,A,bB,aB
    print(f"== ex{n} axis {axis} mkbg {mkbg} tpbg {tpbg} obg {obg}")
    # markers: non-bg pixels in mk
    mrows,mcols=np.where(mk!=mkbg)
    print(" markers:", sorted(Counter(mk[mrows,mcols]).items()), "n=",len(mrows))
    for r,c in zip(mrows,mcols):
        print("   mk",r,c,"=",mk[r,c])
    # template non-bg
    trows,tcols=np.where(tp!=tpbg)
    print(" tp colors", sorted(Counter(tp[trows,tcols]).items()))
    print(" tp bbox rows",trows.min(),trows.max(),"cols",tcols.min(),tcols.max())
