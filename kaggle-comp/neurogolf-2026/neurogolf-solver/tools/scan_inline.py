import sys, json, os, tempfile; sys.path.insert(0,"tools")
from inline import inline_and_clean
import ng, onnx
best=json.load(open('/tmp/best_costs.json'))
bestmap={r['task']:r for r in best}
out=[]
for t in range(1,401):
    src=f"public_open_6645/task{t:03d}.onnx"
    if not os.path.exists(src): continue
    try:
        m=inline_and_clean(src)
        tmp=f"/tmp/inl_{t:03d}.onnx"; onnx.save(m,tmp)
        ap,p,n,c,pa,me=ng.verify_and_cost(tmp,t,profile_runs=4)
        os.remove(tmp)
    except Exception as e:
        ap,p,n,c=False,0,0,None
        print(f"task{t:03d} ERR {str(e)[:60]}", flush=True)
        out.append({"task":t,"ok":False,"err":str(e)[:80]}); continue
    bm=bestmap.get(t,{})
    bc=bm.get('cost'); bp=bm.get('pts',0)
    npts=ng.task_points(c) if ap else 0.0
    win = ap and (bc is None or (c is not None and c<bc) or not bm.get('pass',True))
    out.append({"task":t,"pass":ap,"n":f"{p}/{n}","cost":c,"pts":round(npts,3),
                "best_cost":bc,"best_pts":bp,"win":win})
    flag="WIN" if win else ""
    print(f"task{t:03d} pass={ap} {p}/{n} cost={c} pts={npts:.2f} (best {bc}/{bp}) {flag}", flush=True)
json.dump(out, open('/tmp/inline_scan.json','w'), indent=1)
wins=[o for o in out if o.get('win')]
print("WINS:", len(wins), "extra pts:", round(sum(o['pts']-(o['best_pts'] or 0) for o in wins),2))
