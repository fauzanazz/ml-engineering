"""task133 exact numpy oracle (267/267). Reference for ONNX construction."""
import json, numpy as np
from pathlib import Path
from collections import Counter
DATA=Path(__file__).resolve().parents[2]/"data"
def comps(g,pred):
    seen=np.zeros(g.shape,bool); out=[]; H,W=g.shape
    for r in range(H):
        for c in range(W):
            if pred(g[r,c]) and not seen[r,c]:
                st=[(r,c)];seen[r,c]=True;cc=[]
                while st:
                    y,x=st.pop();cc.append((y,x))
                    for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and pred(g[ny,nx]):
                            seen[ny,nx]=True;st.append((ny,nx))
                out.append(cc)
    return out
def solve(gi):
    g=gi.copy(); H,W=g.shape
    objs=comps(g, lambda v:v!=0)
    if len(objs)<2: return None
    oc=[Counter(int(g[y,x]) for y,x in o) for o in objs]
    cc=Counter()
    for s in oc:
        for c in s: cc[c]+=1
    shared=[c for c,n in cc.items() if n==len(objs)]
    if len(shared)!=1: return None
    M=shared[0]; info=[]
    for o in objs:
        mk=[(y,x) for y,x in o if g[y,x]==M]; sh=[(y,x) for y,x in o if g[y,x]!=M]
        if not mk or not sh: return None
        S=g[sh[0]]; mr=np.array(mk)
        r0,r1,c0,c1=mr[:,0].min(),mr[:,0].max(),mr[:,1].min(),mr[:,1].max()
        n=r1-r0+1
        if (c1-c0+1)!=n or len(mk)!=n*n: return None
        info.append(dict(S=S,mk0=(r0,c0),n=n,sh=sh,shn=len(sh)))
    cand=[t for t in info if t['n']==1]
    if not cand: return None
    ref=max(cand,key=lambda t:t['shn']); rm=ref['mk0']
    glyph=[(y-rm[0],x-rm[1]) for y,x in ref['sh']]
    out=g.copy()
    for t in info:
        if t is ref: continue
        for y,x in t['sh']: out[y,x]=0
        n=t['n']; br,bc=t['mk0']
        for dy,dx in glyph:
            for yy in range(n):
                for xx in range(n):
                    ry,rx=br+dy*n+yy,bc+dx*n+xx
                    if 0<=ry<H and 0<=rx<W: out[ry,rx]=t['S']
    return out
if __name__=="__main__":
    d=json.loads((DATA/'task133.json').read_text())
    A=d['train']+d['test']+d['arc-gen']
    ok=sum(1 for e in A if (lambda p,go: p is not None and p.shape==go.shape and np.array_equal(p,go))(solve(np.array(e['input'])), np.array(e['output'])))
    print(f"task133 oracle {ok}/{len(A)}")
