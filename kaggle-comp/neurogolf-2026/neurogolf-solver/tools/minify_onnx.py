"""Minify ONNX tensor/node names deterministically, preserving input/output names."""
import sys, onnx

def minify(src,dst):
    m=onnx.load(src)
    keep={'input','output'}
    mapping={}
    counter=0
    def nn(name):
        nonlocal counter
        if not name or name in keep: return name
        if name not in mapping:
            mapping[name]=f'n{counter}'
            counter+=1
        return mapping[name]
    m.doc_string=''; m.producer_name=''; m.producer_version=''; m.domain=''; m.model_version=0
    m.graph.name='g'; m.graph.doc_string=''
    for vi in list(m.graph.input)+list(m.graph.output)+list(m.graph.value_info):
        vi.name=nn(vi.name); vi.doc_string=''
    for init in m.graph.initializer: init.name=nn(init.name)
    for n in m.graph.node:
        n.name=''
        n.input[:] = [nn(x) for x in n.input]
        n.output[:] = [nn(x) for x in n.output]
        n.doc_string=''
    onnx.save(m,dst)
    return counter
if __name__=='__main__': print(minify(sys.argv[1],sys.argv[2]))
