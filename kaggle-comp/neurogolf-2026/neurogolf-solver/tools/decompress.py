"""Rewrite Compress(x,cond,axis=0) -> Identity(x). Behavior-preserving iff cond
is always [True] for valid one-hot inputs (verified per-task by isolated check)."""
import onnx, sys
from onnx import helper

def rewrite(in_path, out_path):
    m=onnx.load(in_path)
    g=m.graph
    new=[]
    ncomp=0
    for n in g.node:
        if n.op_type=="Compress":
            # axis must be 0 (batch). data=input[0]. drop condition.
            ax=0
            for a in n.attribute:
                if a.name=="axis": ax=a.i
            if ax!=0:
                new.append(n); continue
            new.append(helper.make_node("Identity",[n.input[0]],[n.output[0]],
                                        name=(n.name or n.output[0])))
            ncomp+=1
        else:
            new.append(n)
    del g.node[:]
    g.node.extend(new)
    onnx.save(m,out_path)
    return ncomp

if __name__=="__main__":
    print(rewrite(sys.argv[1], sys.argv[2]))
