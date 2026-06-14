import sys; sys.path.insert(0, ".")
import numpy as np
from onnx import TensorProto
from builders.common import G

g = G()
g.init("ax_c", np.array([1], np.int64))
g.init("c5s", np.array([5],np.int64)); g.init("c5e", np.array([6],np.int64))
m5 = g.node("Slice", ["input","c5s","c5e","ax_c"], ["m5"])
pres = g.node("MaxPool", [m5], ["pres"], kernel_shape=[3,3], strides=[3,3])  # [1,1,10,10]
E=np.zeros((30,10),np.float32)
for i in range(30): E[i,i//3]=1.0
g.init("E",E); g.init("ET",E.T.copy())
t1=g.node("MatMul",["E","pres"],["t1"])
exp=g.node("MatMul",["t1","ET"],["exp"])
g.init("half",np.array(0.5,np.float32))
out1b=g.node("Greater",[exp,"half"],["out1b"])
gs=g.node("ReduceSum",["input","ax_c"],["gs"],keepdims=1)
ingrid=g.node("Greater",[gs,"half"],["ingrid"])
out1in=g.node("And",[out1b,ingrid],["out1in"])
out1=g.node("Cast",[out1in],["out1"],to=TensorProto.FLOAT)
not1=g.node("Not",[out1in],["not1"])
out0b=g.node("And",[ingrid,not1],["out0b"])
out0=g.node("Cast",[out0b],["out0"],to=TensorProto.FLOAT)
g.init("zeros",np.zeros((1,1,30,30),np.float32))
g.node("Identity",["zeros"],["z"])
outs=[ (out0 if i==0 else out1 if i==1 else "z") for i in range(10)]
g.node("Concat",outs,["output"],axis=1)
g.save("builders/task317.onnx","task317")
import os; print("317",os.path.getsize("builders/task317.onnx"))
