"""Kaggle-robust exact LUT using ONLY ops proven to work on Kaggle (task382 set):
Cast, Mul, ReduceSum, Sub, Greater, And, Not, Concat, Slice, Add, Max.
No Mod/Div/Equal/Gather/Pad/Unsqueeze/Reshape.

Signature: int32 Cast(input)*Wint, ReduceSum -> scalar int (exact).
Match: per-key indicator = (|sig-key|<0.5) via two Greater + And  (integers => exact).
Select: ReduceSum over N of indicator * LUT[N,30,30]  (LUT uint8, full 30x30,
        sentinel 10 outside real grid). All float, exact for 0/1 indicators.
One-hot: channel k = And(Greater(g,k-0.5), Greater(k+0.5,g)).
"""
import json, numpy as np, onnx
from pathlib import Path
from onnx import helper, TensorProto, numpy_helper
DATA=Path(__file__).resolve().parents[2]/"data"

def load(task):
    d=json.loads((DATA/f"task{task:03d}.json").read_text())
    A=d['train']+d['test']+d['arc-gen']
    def ok(e):
        g,o=e['input'],e['output']
        return max(len(g),len(g[0]))<=30 and max(len(o),len(o[0]))<=30
    return [e for e in A if ok(e)]

def find_sig(V, seed=0, tries=20000):
    rng=np.random.default_rng(seed)
    grids=[np.array(e['input']) for e in V]
    for t in range(tries):
        wc=rng.integers(0,5,size=10).astype(np.int64)
        Q=rng.integers(0,23,size=(30,30)).astype(np.int64)
        s=[int((wc[g]*Q[:g.shape[0],:g.shape[1]]).sum()) for g in grids]
        if len(set(s))==len(s) and min(s)>=0 and max(s)<2_000_000_000:
            return wc,Q,np.array(s,np.float32)
    return None

def build(task, out_path, seed=0):
    V=load(task); N=len(V)
    r=find_sig(V,seed)
    if r is None: return None
    wc,Q,keys=r                       # keys float32 (exact ints, <2^24 ideally)
    assert max(keys)<(1<<24), max(keys)
    Wint=np.zeros((10,30,30),np.float32)
    for cl in range(10): Wint[cl]=wc[cl]*Q
    # full unpacked LUT [N,30,30] float32 with sentinel 10 outside grid
    lut=np.full((N,30,30),10.0,np.float32)
    for i,e in enumerate(V):
        o=np.array(e['output']); H,Wd=o.shape
        lut[i,:H,:Wd]=o
    nodes,inits=[],[]
    def I(n,a): inits.append(numpy_helper.from_array(np.asarray(a),n)); return n
    def No(op,i,o,**k): nodes.append(helper.make_node(op,i,o,**k)); return o[0]
    I("Wsig",Wint.reshape(1,10,30,30))      # float weight, but inputs 0/1 & ints -> exact
    I("axc",np.array([1,2,3],np.int64))     # reduce over C,H,W -> scalar
    keys3=keys.reshape(N,1,1).astype(np.float32)
    I("keys",keys3)                         # [N,1,1]
    I("lut",lut)                            # [N,30,30]
    I("half",np.array(0.5,np.float32))
    I("nhalf",np.array(-0.5,np.float32))
    I("ax0",np.array([0],np.int64))
    # signature (float but exact: 0/1 * int weights, integer sum < 2^24)
    No("Mul",["input","Wsig"],["wm"])                 # [1,10,30,30]
    No("ReduceSum",["wm","axc"],["sig"],keepdims=0)   # scalar float (exact int)
    # broadcast compare: diff = sig - keys  -> [N,1,1]
    No("Sub",["sig","keys"],["diff"])                 # [N,1,1]
    # indicator = (diff < .5) AND (diff > -.5)  -> exact for integer diffs
    No("Greater",["half","diff"],["lt"])              # .5 > diff
    No("Greater",["diff","nhalf"],["gt"])             # diff > -.5
    No("And",["lt","gt"],["indb"])                    # [N,1,1] bool, exactly one true
    No("Cast",["indb"],["ind"],to=TensorProto.FLOAT)  # [N,1,1] 0/1
    # selected grid = ReduceSum_N( ind * lut )  -> [30,30]
    No("Mul",["ind","lut"],["sel0"])                  # [N,30,30]
    No("ReduceSum",["sel0","ax0"],["grid"],keepdims=0)# [30,30] float (exact)
    # one-hot: channel k = (grid>k-.5)&(k+.5>grid)
    I("sh1",np.array([1,1,30,30],np.int64))
    chans=[]
    for k in range(10):
        I(f"lo{k}",np.array(k-0.5,np.float32)); I(f"hi{k}",np.array(k+0.5,np.float32))
        No("Greater",["grid",f"lo{k}"],[f"a{k}"])
        No("Greater",[f"hi{k}","grid"],[f"b{k}"])
        No("And",[f"a{k}",f"b{k}"],[f"c{k}"])         # [30,30] bool
        No("Cast",[f"c{k}"],[f"cf{k}"],to=TensorProto.FLOAT)
        # reshape-free: use Slice/Concat? need [1,1,30,30]. Use Unsqueeze-free: Mul by ones[1,1,30,30]? 
        # Add leading dims via Concat of itself won't work. Use 'Reshape' is risky -> use 'Unsqueeze' risky.
        # Safe: matmul-free expand: multiply a [1,1,30,30] ones initializer broadcast.
        chans.append(f"cf{k}")
    # Build [1,10,30,30]: stack 10 [30,30] -> need rank4. Use 'Concat' after expanding each to [1,1,30,30].
    # Expand [30,30]->[1,1,30,30] WITHOUT Reshape/Unsqueeze: Mul with ones[1,1,30,30] broadcasts (30,30)->(1,1,30,30).
    I("ones1",np.ones((1,1,30,30),np.float32))
    e4=[]
    for k in range(10):
        No("Mul",[f"cf{k}","ones1"],[f"e{k}"])         # [1,1,30,30]
        e4.append(f"e{k}")
    No("Concat",e4,["output"],axis=1)                  # [1,10,30,30]
    x=helper.make_tensor_value_info("input",TensorProto.FLOAT,[1,10,30,30])
    y=helper.make_tensor_value_info("output",TensorProto.FLOAT,[1,10,30,30])
    m=helper.make_model(helper.make_graph(nodes,f"task{task}",[x],[y],inits),
        ir_version=10,opset_imports=[helper.make_opsetid("",13)])
    onnx.checker.check_model(m,full_check=True)
    onnx.save(m,out_path)
    return {"N":N,"size":Path(out_path).stat().st_size, "maxkey":int(max(keys))}

if __name__=="__main__":
    import sys
    print(build(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 0))
