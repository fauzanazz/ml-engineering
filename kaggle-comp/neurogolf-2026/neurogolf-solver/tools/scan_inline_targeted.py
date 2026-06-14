import sys, json, os; sys.path.insert(0,"tools")
from inline import inline_and_clean
import ng, onnx
best={r['task']:r for r in json.load(open('/tmp/best_costs.json'))}
# target: expensive best tasks (cost>80000) + confirmed-fail tasks
targets=sorted([t for t,r in best.items() if (r.get('cost') or 0)>80000 or not r['pass']])
print("targets", len(targets), flush=True)
out=[]
for t in targets:
    src=f"public_open_6645/task{t:03d}.onnx"
    if not os.path.exists(src):
        continue
    try:
        m=inline_and_clean(src); tmp=f"/tmp/it_{t}.onnx"; onnx.save(m,tmp)
        ap,p,n,c,pa,me=ng.verify_and_cost(tmp,t,profile_runs=3); os.remove(tmp)
    except Exception as e:
        print(f"task{t:03d} ERR {str(e)[:50]}", flush=True); continue
    bm=best[t]; bc=bm.get('cost'); bp=bm.get('pts',0); bpass=bm['pass']
    npts=ng.task_points(c) if ap else 0.0
    win = ap and (not bpass or (c is not None and bc is not None and c<bc))
    gain = (npts-bp) if win else 0
    out.append({"task":t,"pass":ap,"cost":c,"pts":round(npts,3),"best_cost":bc,"best_pts":bp,"win":win,"gain":round(gain,3)})
    print(f"task{t:03d} pass={ap} {p}/{n} cost={c} pts={npts:.2f} best={bc}/{bp:.2f} {'WIN +%.2f'%gain if win else ''}", flush=True)
json.dump(out, open('/tmp/inline_targeted.json','w'), indent=1)
wins=[o for o in out if o['win']]
print("WINS",len(wins),"total gain", round(sum(o['gain'] for o in wins),2))
