"""Add static value_info for node outputs from ORT profile runtime shapes."""
import sys, os, json, onnx, numpy as np, onnxruntime as ort
from onnx import helper, TensorProto

def staticize(src, dst, sample_input=None):
    m=onnx.load(src)
    for _n in m.graph.node:
        if _n.output: _n.name=_n.output[0]
    opts=ort.SessionOptions(); opts.enable_profiling=True
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess=ort.InferenceSession(m.SerializeToString(), opts, providers=['CPUExecutionProvider'])
    if sample_input is None: sample_input=np.zeros((1,10,30,30),np.float32); sample_input[0,0,0,0]=1
    sess.run(['output'], {'input': sample_input})
    trace=sess.end_profiling()
    events=json.load(open(trace)); os.remove(trace)
    # map node names to outputs, using original names
    node_outputs={n.name:list(n.output) for n in m.graph.node if n.name}
    # also after sanitize? profile node names are usually node.name + _kernel_time
    existing={v.name for v in list(m.graph.input)+list(m.graph.value_info)+list(m.graph.output)}
    # infer dtypes from shape-inferred graph where possible, fallback float
    try:
        inf=onnx.shape_inference.infer_shapes(m, strict_mode=False)
        dtype={v.name:v.type.tensor_type.elem_type for v in list(inf.graph.input)+list(inf.graph.value_info)+list(inf.graph.output) if v.type.HasField('tensor_type')}
    except Exception:
        dtype={}
    added=0
    for ev in events:
        if ev.get('cat')!='Node': continue
        args=ev.get('args',{})
        shapes=args.get('output_type_shape')
        if not shapes: continue
        nn=ev.get('name','').replace('_kernel_time','')
        outs=node_outputs.get(nn)
        if not outs: continue
        for i,sd in enumerate(shapes):
            if i>=len(outs): continue
            out=outs[i]
            if not out or out in existing or out=='output': continue
            # sd maps dtype string -> dims, take first
            dims=list(next(iter(sd.values())))
            # dtype from event key
            type_key=next(iter(sd.keys()))
            elem=dtype.get(out)
            if elem is None:
                elem={'float':TensorProto.FLOAT,'int32':TensorProto.INT32,'int64':TensorProto.INT64,'bool':TensorProto.BOOL,'uint8':TensorProto.UINT8}.get(type_key, TensorProto.FLOAT)
            m.graph.value_info.append(helper.make_tensor_value_info(out, elem, dims))
            existing.add(out); added+=1
    onnx.save(m,dst)
    return added
if __name__=='__main__': print(staticize(sys.argv[1], sys.argv[2]))
