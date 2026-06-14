import sys; sys.path.insert(0, ".")
import numpy as np
from onnx import TensorProto
from builders.common import G

g = G()
# slice only channel 8
g.init("ax_c", np.array([1], np.int64))
g.init("c8s", np.array([8], np.int64)); g.init("c8e", np.array([9], np.int64))
in8 = g.node("Slice", ["input","c8s","c8e","ax_c"], ["in8"])
# shift down 1 row via Pad top then slice
g.init("pads_down", np.array([0,0,1,0, 0,0,0,0], np.int64))
g.init("zf", np.array(0, np.float32))
shp = g.node("Pad", [in8,"pads_down","zf"], ["shp"], mode="constant")
g.init("s0", np.array([0], np.int64)); g.init("s30", np.array([30], np.int64)); g.init("ax_h", np.array([2], np.int64))
out2 = g.node("Slice", ["shp","s0","s30","ax_h"], ["out2"])  # float [1,1,30,30]
# in-grid mask (bool)
gs = g.node("ReduceSum", ["input","ax_c"], ["gs"], keepdims=1)
g.init("half", np.array(0.5, np.float32))
ingrid = g.node("Greater", [gs,"half"], ["ingrid"])
is2 = g.node("Greater", [out2,"half"], ["is2"])
not2 = g.node("Not", [is2], ["not2"])
out0b = g.node("And", [ingrid,not2], ["out0b"])
out0 = g.node("Cast", [out0b], ["out0"], to=TensorProto.FLOAT)
# zeros [1,1,30,30] as a constant initializer-derived tensor (reused)
g.init("zeros", np.zeros((1,1,30,30), np.float32))
g.node("Identity", ["zeros"], ["z"])
outs = []
for i in range(10):
    outs.append(out0 if i==0 else out2 if i==2 else "z")
g.node("Concat", outs, ["output"], axis=1)
g.save("builders/task261.onnx", "task261")
import os; print("saved", os.path.getsize("builders/task261.onnx"))
