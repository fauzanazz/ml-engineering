import sys, types, json
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
t=int(sys.argv[1]); d=sys.argv[2]
DATA=Path(__file__).resolve().parents[2]/"data"
for n in ['IPython','IPython.display','matplotlib','matplotlib.pyplot','onnx_tool']:
    sys.modules[n]=types.ModuleType(n)
sys.modules['IPython'].display=sys.modules['IPython.display']
sys.modules['IPython.display'].display=lambda *a,**k:None
sys.modules['IPython.display'].FileLink=lambda *a,**k:None
sys.modules['matplotlib'].pyplot=sys.modules['matplotlib.pyplot']
sys.modules['onnx_tool'].model_profile=lambda *a,**k:None
sys.path.insert(0,str(DATA/'neurogolf_utils'))
import neurogolf_utils as ng
ng._NEUROGOLF_DIR=str(DATA.resolve())+'/'
ex=ng.load_examples(t); allex=ex['train']+ex['test']+ex['arc-gen']
try:
    m=ng.sanitize_model(onnx.load(f'{d}/task{t:03d}.onnx'))
    o=ort.SessionOptions(); o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s=ort.InferenceSession(m.SerializeToString(),o,providers=['CPUExecutionProvider'])
except Exception as e:
    print(json.dumps({"task":t,"pass":False,"err":str(e)[:60]})); sys.exit()
p=tot=0
for e in allex:
    b=ng.convert_to_numpy(e)
    if not b: continue
    tot+=1
    try:
        out=s.run(['output'],{'input':b['input']})[0]
        p+=int(np.array_equal((out>0.0).astype(float),b['output']))
    except Exception: pass
print(json.dumps({"task":t,"pass":p==tot and tot>0,"n":f"{p}/{tot}"}))
