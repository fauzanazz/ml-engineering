import sys, json; sys.path.insert(0,"tools")
import ng
rows=[]
for t in range(1,401):
    ap,p,n,c,pa,me=ng.verify_and_cost(f"/tmp/best/task{t:03d}.onnx",t,profile_runs=2)
    pts=ng.task_points(c) if ap else 0.0
    rows.append({"task":t,"pass":ap,"n":f"{p}/{n}","cost":c,"pts":round(pts,3)})
    if not ap:
        print(f"task{t:03d} FAIL {p}/{n} cost={c}", flush=True)
json.dump(rows, open('/tmp/best2.json','w'))
tot=sum(r['pts'] for r in rows)
fails=[r['task'] for r in rows if not r['pass']]
print("GENUINE FAILS:", fails)
print("LOCAL TOTAL (profile_runs=2):", round(tot,2))
