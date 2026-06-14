import sys, os, json, math, onnx, collections
sys.path.insert(0,'tools')
from staticize_value_info import staticize
import ng
from onnx import shape_inference, AttributeProto
SRC='scratch/public/a6335_dc'; OUT='scratch/a6335_dc_staticized'; BASE='scratch/blend_t133_static'; os.makedirs(OUT,exist_ok=True)
# candidates: clean but dyn from previous scan
scan_path='scratch/a6335_scan.json' if os.path.exists('scratch/a6335_scan.json') else 'scratch/a6335_dc_scan.json'
rows=json.load(open(scan_path)) if os.path.exists(scan_path) else []
cands=[r['task'] for r in rows if r.get('valid') in ('dynshape','dyn','noshape')]
if not cands:
 cands=[133]
def valid(path):
 try: m=onnx.load(path); g=shape_inference.infer_shapes(m, strict_mode=True).graph
 except Exception as e: return 'infer'
 EX={'LOOP','SCAN','NONZERO','UNIQUE','SCRIPT','FUNCTION','COMPRESS'}
 if any(n.op_type.upper() in EX or 'Sequence' in n.op_type for n in m.graph.node): return 'ban'
 if any(o.domain not in ('','ai.onnx') for o in m.opset_import): return 'dom'
 if m.functions: return 'func'
 tn=set()
 for nd in g.node:
  for a in nd.attribute:
   if a.type in (AttributeProto.GRAPH,AttributeProto.GRAPHS): return 'sub'
  for o in nd.output:
   if o: tn.add(o)
 tmap={t.name:t for t in list(g.input)+list(g.value_info)+list(g.output)}; tn.update(tmap)
 for x in tn:
  it=tmap.get(x)
  if not it: return 'missing'
  if not it.type.HasField('tensor_type'): continue
  tt=it.type.tensor_type
  if not tt.HasField('shape'): return 'noshape'
  for dm in tt.shape.dim:
   if dm.HasField('dim_param') or not dm.HasField('dim_value') or dm.dim_value<=0: return 'dyn'
 return 'OK'
def is_lut(path,nex):
 m=onnx.load(path); c=collections.Counter(n.op_type for n in m.graph.node)
 return max(c.get('Equal',0),c.get('ScatterElements',0),c.get('Gather',0))>=max(20,nex*0.6)
res=[]
for t in cands:
 src=f'{SRC}/task{t:03d}.onnx'; dst=f'{OUT}/task{t:03d}.onnx'
 if not os.path.exists(src): continue
 try:
  # use first train input as profile sample
  import neurogolf_utils as off
 except Exception: pass
 task=ng.examples(t); examples=task['train']+task['test']+task['arc-gen']; sample=None
 for e in examples:
  b=ng.ng.convert_to_numpy(e)
  if b is not None: sample=b['input']; break
 try:
  added=staticize(src,dst,sample)
 except Exception as e:
  res.append({'task':t,'err':str(e)[:60]}); continue
 v=valid(dst); rec={'task':t,'added':added,'valid':v}
 if v=='OK':
  ap,p,n,c,pa,me=ng.verify_and_cost(dst,t,profile_runs=2)
  base_path=f'{BASE}/task{t:03d}.onnx'
  if os.path.exists(base_path):
   base_pass,_,_,bc,_,_=ng.verify_and_cost(base_path,t,profile_runs=2)
  else:
   base_pass=False; bc=None
  bpts=max(1.0,25.0-math.log(max(1.0,bc))) if base_pass and bc else 0.0
  npts=ng.task_points(c) if ap and c else 0.0
  rec.update(pass_=ap,n=f'{p}/{n}',cost=c,pts=round(npts,3),base_pts=round(bpts,3),lut=is_lut(dst,len(examples)),gain=round(npts-bpts,3))
  if ap and c and npts>bpts+0.05 and not rec['lut']:
   print('WIN',rec,flush=True)
 res.append(rec)
json.dump(res,open('scratch/a6335_staticize_scan.json','w'),indent=1)
w=[r for r in res if r.get('pass_') and r.get('gain',0)>0.05 and not r.get('lut')]
print('cands',len(cands),'validOK',sum(1 for r in res if r.get('valid')=='OK'),'wins',len(w),'gain',round(sum(r['gain'] for r in w),3))
