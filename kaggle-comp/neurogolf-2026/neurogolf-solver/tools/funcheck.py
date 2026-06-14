import sys, types, json
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
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
d=sys.argv[1]
fails=[]
for t in range(1,401):
    ex=ng.load_examples(t)
    try:
        san=ng.sanitize_model(onnx.load(f'{d}/task{t:03d}.onnx'))
        s=ort.InferenceSession(san.SerializeToString(),providers=['CPUExecutionProvider'])
    except Exception as e:
        fails.append((t,'load',str(e)[:40])); print(f"task{t:03d} LOADFAIL", flush=True); continue
    allex=ex['train']+ex['test']+ex['arc-gen']
    p=tot=0
    for e in allex:
        b=ng.convert_to_numpy(e)
        if not b: continue
        tot+=1
        try:
            o=s.run(['output'],{'input':b['input']})[0]
            p+=int(np.array_equal((o>0.0).astype(float),b['output']))
        except Exception: pass
    if p!=tot or tot==0:
        fails.append((t,f"{p}/{tot}")); print(f"task{t:03d} FAIL {p}/{tot}", flush=True)
print("GENUINE FAILS:", fails)
json.dump(fails, open('/tmp/genuine_fails.json','w'))
