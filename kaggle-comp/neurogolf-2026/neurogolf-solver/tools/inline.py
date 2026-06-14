import onnx, onnx.inliner
from onnx import helper

def inline_and_clean(path):
    m=onnx.load(path)
    m=onnx.inliner.inline_local_functions(m)
    # drop non-standard opset imports
    keep=[o for o in m.opset_import if o.domain in ("","ai.onnx")]
    has_default=any(o.domain in ("","ai.onnx") for o in keep)
    del m.opset_import[:]
    if not has_default:
        keep=[helper.make_opsetid("",17)]
    m.opset_import.extend(keep)
    del m.functions[:]
    return m
