import sys, json, os; sys.path.insert(0,"tools")
from onnx import shape_inference, AttributeProto
import onnx, ng
clean=json.load(open("/tmp/a6284_clean.json"))
best={int(json.loads(l)['task']):json.loads(l) for l in open('/tmp/iso_best.jsonl')}
best2={r['task']:r for r in json.load(open('/tmp/best2.json'))}
def valid(path):
    m=onnx.load(path)
    if m.functions: return "func"
    for o in m.opset_import:
        if o.domain not in ("","ai.onnx"): return "domain:"+o.domain
    try: onnx.checker.check_model(m, full_check=True)
    except Exception as e: return "checker"
    try: g=shape_inference.infer_shapes(m, strict_mode=True).graph
    except Exception: return "infer"
    tn=set()
    for nd in g.node:
        for a in nd.attribute:
            if a.type in (AttributeProto.GRAPH,AttributeProto.GRAPHS): return "subgraph"
        for o in nd.output:
            if o: tn.add(o)
    tmap={t.name:t for t in list(g.input)+list(g.value_info)+list(g.output)}
    tn.update(tmap)
    for x in tn:
        it=tmap.get(x)
        if not it: return "missing"
        if it.type.HasField("sequence_type"): return "seq"
        if not it.type.HasField("tensor_type"): continue
        tt=it.type.tensor_type
        if not tt.HasField("shape"): return "noshape"
        for dm in tt.shape.dim:
            if dm.HasField("dim_param") or not dm.HasField("dim_value") or dm.dim_value<=0:
                return "dynshape"
    return "OK"
out=[]
for t in clean:
    p=f"/tmp/dl/a6284/task{t:03d}.onnx"
    v=valid(p)
    rec={"task":t,"valid":v}
    if v=="OK":
        ap,pp,nn,c,pa,me=ng.verify_and_cost(p,t,profile_runs=2)
        bp = best.get(t,{}).get('pass',False)
        bc = best2.get(t,{}).get('cost')
        bpts = best2.get(t,{}).get('pts',0) if bp else 0.0
        npts = ng.task_points(c) if (ap and c is not None) else 0.0
        rec.update(pass_=ap, n=f"{pp}/{nn}", cost=c, pts=round(npts,3),
                   best_pass=bp, best_cost=bc, best_pts=round(bpts,3),
                   win=bool(ap and c is not None and npts>bpts+0.05))
        if rec["win"]:
            print(f"WIN task{t:03d} cost {bc}->{c} pts {bpts:.2f}->{npts:.2f} +{npts-bpts:.2f}", flush=True)
    out.append(rec)
json.dump(out, open("/tmp/a6284_scan.json","w"), indent=1)
wins=[r for r in out if r.get("win")]
print("VALID-OK:", sum(1 for r in out if r['valid']=='OK'), "WINS:", len(wins),
      "gain", round(sum(r['pts']-r['best_pts'] for r in wins),2))
