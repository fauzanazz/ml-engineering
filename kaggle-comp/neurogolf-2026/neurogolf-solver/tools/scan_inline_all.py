import sys, json, os; sys.path.insert(0,"tools")
from inline import inline_and_clean
import ng, onnx
best={r['task']:r for r in json.load(open('/tmp/best_costs.json'))}
done={133,205,230,261,282,317,366,370,382}
out=[]
for t in range(1,401):
    if t in done: continue
    src=f"public_open_6645/task{t:03d}.onnx"
    if not os.path.exists(src): continue
    try:
        m=inline_and_clean(src)
        # quick skip: if not golf-derived & node count huge, still test
        tmp=f"/tmp/ia_{t}.onnx"; onnx.save(m,tmp)
        ap,p,n,c,pa,me=ng.verify_and_cost(tmp,t,profile_runs=2); os.remove(tmp)
    except Exception as e:
        continue
    bm=best[t]; bc=bm.get('cost'); bp=bm.get('pts',0); bpass=bm['pass']
    npts=ng.task_points(c) if ap else 0.0
    win = ap and (not bpass or (c is not None and bc is not None and c<bc and npts>bp+0.05))
    if win:
        g=npts-(bp if bpass else 0)
        out.append({"task":t,"cost":c,"pts":round(npts,3),"best_cost":bc,"best_pts":bp,"gain":round(g,3)})
        print(f"WIN task{t:03d} cost {bc}->{c} pts {bp:.2f}->{npts:.2f} +{g:.2f}", flush=True)
json.dump(out, open('/tmp/inline_all_wins.json','w'), indent=1)
print("EXTRA WINS", len(out), "gain", round(sum(o['gain'] for o in out),2))
