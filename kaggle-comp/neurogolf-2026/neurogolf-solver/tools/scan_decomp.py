import sys, json, os, glob; sys.path.insert(0,"tools")
from decompress import rewrite
from onnx import shape_inference, AttributeProto
import onnx, ng
SRC="/tmp/dl/a6335"
OUT="/tmp/a6335_dc"; os.makedirs(OUT,exist_ok=True)
best={int(json.loads(l)['task']):json.loads(l) for l in open('/tmp/iso_best.jsonl')}
best2={r['task']:r for r in json.load(open('/tmp/best2.json'))}
def valid(path):
    m=onnx.load(path)
    if m.functions: return "func"
    for o in m.opset_import:
        if o.domain not in ("","ai.onnx"): return "dom"
    EX={"LOOP","SCAN","NONZERO","UNIQUE","SCRIPT","FUNCTION","COMPRESS"}
    for n in m.graph.node:
        if n.op_type.upper() in EX or "Sequence" in n.op_type: return "ban:"+n.op_type
    try: onnx.checker.check_model(m, full_check=True)
    except Exception: return "checker"
    try: gg=shape_inference.infer_shapes(m, strict_mode=True).graph
    except Exception: return "infer"
    tn=set()
    for nd in gg.node:
        for a in nd.attribute:
            if a.type in (AttributeProto.GRAPH,AttributeProto.GRAPHS): return "sub"
        for o in nd.output:
            if o: tn.add(o)
    tmap={t.name:t for t in list(gg.input)+list(gg.value_info)+list(gg.output)}
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
                return "dyn"
    return "OK"
import collections
def is_lut(path, nex):
    m=onnx.load(path); c=collections.Counter(n.op_type for n in m.graph.node)
    # per-example memorization heuristic: many Equal/Gather/Scatter ~ #examples
    susp=max(c.get("Equal",0),c.get("ScatterElements",0),c.get("Gather",0))
    return susp>=max(20, nex*0.6)
res=[]
files=sorted(glob.glob(SRC+"/task*.onnx"))
for f in files:
    b=os.path.basename(f)
    if b=="task000.onnx": continue
    t=int(b[4:7])
    m=onnx.load(f)
    has_c=any(n.op_type=="Compress" for n in m.graph.node)
    if not has_c: continue
    dst=f"{OUT}/task{t:03d}.onnx"
    try:
        rewrite(f,dst)
    except Exception as e:
        res.append({"task":t,"err":str(e)[:60]}); continue
    v=valid(dst)
    rec={"task":t,"valid":v}
    if v=="OK":
        exl=ng.examples(t); nex=len(exl)
        rec["lut"]=is_lut(dst,nex)
        ap,pp,nn,c,pa,me=ng.verify_and_cost(dst,t,profile_runs=2)
        bp=best.get(t,{}).get('pass',False)
        bc=best2.get(t,{}).get('cost'); bpts=best2.get(t,{}).get('pts',0) if bp else 0.0
        npts=ng.task_points(c) if (ap and c is not None) else 0.0
        rec.update(pass_=ap,n=f"{pp}/{nn}",cost=c,pts=round(npts,3),
                   best_cost=bc,best_pts=round(bpts,3),
                   win=bool(ap and c is not None and (not rec["lut"]) and npts>bpts+0.05))
        if rec.get("win"):
            print(f"WIN task{t:03d} cost {bc}->{c} pts {bpts:.2f}->{npts:.2f} +{npts-bpts:.2f} lut={rec['lut']}",flush=True)
    res.append(rec)
json.dump(res, open("/tmp/a6335_dc_scan.json","w"), indent=1)
wins=[r for r in res if r.get("win")]
print("processed",len(res),"valid_OK",sum(1 for r in res if r['valid']=='OK'),
      "non_lut_wins",len(wins),"gain",round(sum(r['pts']-r['best_pts'] for r in wins),2))
