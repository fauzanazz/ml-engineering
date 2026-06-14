import json
import numpy as np
from pathlib import Path
DATA=Path(__file__).resolve().parents[2]/"data"
d=json.loads((DATA/"task366.json").read_text())
ex=d['train']+d['test']+d['arc-gen']
def bg(g):
    v,c=np.unique(g,return_counts=True); return v[np.argmax(c)]
shapes=set(); vh=set()
for e in ex:
    gi=np.array(e['input']); go=np.array(e['output'])
    H,W=gi.shape; OH,OW=go.shape
    vh.add((H%2==0,W%2==0,OH==H//2 and OW==W, OW==W//2 and OH==H, H,W,OH,OW))
# summarize
vcnt=sum(1 for e in ex if np.array(e['output']).shape[0]==np.array(e['input']).shape[0]//2 and np.array(e['output']).shape[1]==np.array(e['input']).shape[1])
hcnt=sum(1 for e in ex if np.array(e['output']).shape[1]==np.array(e['input']).shape[1]//2 and np.array(e['output']).shape[0]==np.array(e['input']).shape[0])
both=sum(1 for e in ex if (np.array(e['input']).shape[0]%2==0 and np.array(e['input']).shape[1]%2==0))
print("V-out",vcnt,"H-out",hcnt,"total",len(ex),"both-even-input",both)
# for ambiguous (both even input), see which the output picks
amb=0
for e in ex:
    gi=np.array(e['input']);go=np.array(e['output'])
    H,W=gi.shape
    if H%2==0 and W%2==0 and H==W:
        amb+=1
print("square both-even (truly ambiguous shape):",amb)
# Determine marker half by pixel sparsity
def info(g):
    b=bg(g); return (g!=b).mean()
mism=0
for e in ex:
    gi=np.array(e['input']);go=np.array(e['output'])
    H,W=gi.shape;OH,OW=go.shape
    if OH==H//2 and OW==W:
        A,B=gi[:H//2],gi[H//2:]
    else:
        A,B=gi[:,:W//2],gi[:,W//2:]
    obg=bg(go)
    # marker half = bg matches obg
    aB,bB=bg(A),bg(B)
    mk = A if aB==obg else B
    # is marker half sparser?
    if not (info(mk) <= info(A if mk is B else B)): mism+=1
print("marker-not-sparser cases:",mism)
