import sys, json, os, shutil; sys.path.insert(0,"tools")
from inline import inline_and_clean
import ng, onnx
from pathlib import Path
BLEND="blend_v7"
shutil.rmtree(BLEND, ignore_errors=True)
shutil.copytree("/tmp/best", BLEND)
# inlined wins to take from public_open_6645
inlined_tasks=[133,230,261,317,366,205,370,382]  # 282 use my handbuilt
hand={282:"builders/task282.onnx"}
report=[]
for t in inlined_tasks:
    m=inline_and_clean(f"public_open_6645/task{t:03d}.onnx")
    dst=f"{BLEND}/task{t:03d}.onnx"
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m,dst)
    ap,p,n,c,pa,me=ng.verify_and_cost(dst,t,profile_runs=3)
    report.append((t,"inlined",ap,f"{p}/{n}",c,round(ng.task_points(c) if ap else 0,3)))
for t,src in hand.items():
    dst=f"{BLEND}/task{t:03d}.onnx"
    shutil.copy(src,dst)
    ap,p,n,c,pa,me=ng.verify_and_cost(dst,t,profile_runs=3)
    report.append((t,"hand",ap,f"{p}/{n}",c,round(ng.task_points(c) if ap else 0,3)))
for r in sorted(report): print(r)
ok=all(r[2] for r in report)
print("ALL PASS:", ok, "files:", len(os.listdir(BLEND)))
