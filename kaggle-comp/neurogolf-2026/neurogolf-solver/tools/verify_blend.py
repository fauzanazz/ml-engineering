import sys, json; sys.path.insert(0,"tools")
import ng
swapped=[133,205,230,261,282,317,366,370,382]
tot=0.0; fails=[]
res={}
for t in range(1,401):
    p=f"blend_v7/task{t:03d}.onnx"
    ap,pa,n,c,_,_=ng.verify_and_cost(p,t,profile_runs=2)
    pts=ng.task_points(c) if ap else 0.0
    tot+=pts
    res[t]={"pass":ap,"cost":c,"pts":round(pts,3)}
    if not ap: fails.append((t,f"{pa}/{n}"))
    if t in swapped:
        print(f"task{t:03d} pass={ap} {pa}/{n} cost={c} pts={pts:.3f}", flush=True)
print("FAILS:", fails)
print("PROJECTED TOTAL:", round(tot,2))
json.dump(res, open('/tmp/blend_v7_res.json','w'))
