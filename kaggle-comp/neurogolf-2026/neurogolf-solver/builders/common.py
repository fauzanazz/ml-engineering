"""Shared ONNX builder helpers (cost-minimized recipe from task382)."""
from __future__ import annotations
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

B, C, H, W = 1, 10, 30, 30

class G:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self._seen = set()
    def init(self, name, arr):
        if name in self._seen:
            return name
        self._seen.add(name)
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name
    def node(self, op, ins, outs, **attrs):
        self.nodes.append(helper.make_node(op, ins, outs, **attrs))
        return outs[0] if len(outs) == 1 else outs
    def save(self, path, name="g", opset=13):
        x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [B, C, H, W])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [B, C, H, W])
        m = helper.make_model(
            helper.make_graph(self.nodes, name, [x], [y], self.inits),
            ir_version=10, opset_imports=[helper.make_opsetid("", opset)])
        onnx.checker.check_model(m, full_check=True)
        onnx.save(m, path)
        return m

def channel_slices(g):
    """Return list of 10 single-channel float tensors [1,1,30,30]."""
    g.init("ax_c", np.array([1], np.int64))
    chans = []
    for i in range(C):
        g.init(f"cs{i}", np.array([i], np.int64))
        g.init(f"ce{i}", np.array([i + 1], np.int64))
        chans.append(g.node("Slice", ["input", f"cs{i}", f"ce{i}", "ax_c"], [f"in{i}"]))
    return chans
