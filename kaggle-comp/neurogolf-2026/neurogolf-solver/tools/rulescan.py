"""Detect simple exact rules per task across all examples (train+test+arc-gen)."""
import json, sys, numpy as np
from pathlib import Path
DATA=Path(__file__).resolve().parents[2]/"data"

def ex(t):
    d=json.loads((DATA/f"task{t:03d}.json").read_text())
    return [(np.array(e['input']),np.array(e['output'])) for e in d['train']+d['test']+d['arc-gen']]

def all_true(pairs, fn):
    for gi,go in pairs:
        try:
            p=fn(gi)
        except Exception:
            return False
        if p is None or p.shape!=go.shape or not np.array_equal(p,go): return False
    return True

def detect(t):
    P=ex(t)
    if not P: return None
    same_shape=all(gi.shape==go.shape for gi,go in P)
    rules=[]
    if same_shape:
        if all_true(P, lambda g: g): rules.append("identity")
        for k in (1,2,3):
            for ax in (0,1):
                rules+=[f"flip{ax}"] if all_true(P, lambda g,ax=ax: np.flip(g,ax)) else []
        if all_true(P, lambda g: g.T) and all(gi.shape[0]==gi.shape[1] for gi,_ in P): rules.append("transpose")
        for k in (1,2,3):
            rules+=[f"rot{k}"] if all(gi.shape==go.shape for gi,go in P) and all_true(P, lambda g,k=k: np.rot90(g,k)) else []
        # global recolor (bijection consistent across all)
        m={}
        ok=True
        for gi,go in P:
            for a,b in zip(gi.flat,go.flat):
                if a in m and m[a]!=b: ok=False;break
                m[a]=b
            if not ok:break
        if ok and any(k!=v for k,v in m.items()): rules.append(f"recolor{m}")
        # shifts with zero fill
        for dr in (-2,-1,0,1,2):
            for dc in (-2,-1,0,1,2):
                if dr==0 and dc==0: continue
                def sh(g,dr=dr,dc=dc):
                    o=np.zeros_like(g)
                    H,W=g.shape
                    for r in range(H):
                        for c in range(W):
                            nr,nc=r+dr,c+dc
                            if 0<=nr<H and 0<=nc<W: o[nr,nc]=g[r,c]
                    return o
                if all_true(P, sh): rules.append(f"shift{dr},{dc}")
    out_shapes={tuple(go.shape) for _,go in P}
    in_shapes={tuple(gi.shape) for gi,_ in P}
    return {"task":t,"same_shape":same_shape,"in_shapes":len(in_shapes),"out_shapes":len(out_shapes),"rules":rules}

if __name__=="__main__":
    costs={r['task']:r for r in json.load(open('/tmp/best_costs.json'))}
    hits=[]
    for t in range(1,401):
        d=detect(t)
        if d and d["rules"]:
            c=costs.get(t,{})
            d["cost"]=c.get("cost"); d["pts"]=c.get("pts")
            hits.append(d)
            print(f"task{t:03d} cost={c.get('cost')} pts={c.get('pts')} rules={d['rules']}", flush=True)
    print("TOTAL trivial-rule tasks:", len(hits))
    json.dump(hits, open('/tmp/rulehits.json','w'), indent=1)
