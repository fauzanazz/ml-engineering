import subprocess, sys
try:
    import onnxruntime as _ort  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'onnxruntime'])
    import onnxruntime as _ort  # noqa: F401

import json
import math
import os
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto
from onnx import helper as oh
from onnx import numpy_helper as onh

NUM_TASKS = 400
INPUT_ROOT = Path('/tmp/neurogolf_artifacts')
OUTPUT_ZIP = Path('submission.zip')

AFR1STE_DATASET = 'afr5689'
KONBU17_DATASET = 'konbu'
KONBU17_TASKS = frozenset({205, 308, 370, 382})

COMP_DIR = Path('../data')

def find_onnx_root(slug):
    for entry in INPUT_ROOT.rglob('task001.onnx'):
        if slug in str(entry):
            return entry.parent
    raise FileNotFoundError('Could not locate task ONNX root for ' + slug)

afr1ste_dir = find_onnx_root(AFR1STE_DATASET)
konbu17_dir = find_onnx_root(KONBU17_DATASET)
print('afr1ste source :', afr1ste_dir)
print('konbu17 source :', konbu17_dir)
print('comp data dir  :', COMP_DIR)

def encode_grid(grid):
    arr = np.array(grid, dtype=np.int32)
    h, w = arr.shape
    t = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            v = int(arr[r, c])
            if 0 <= v < 10:
                t[0, v, r, c] = 1.0
    return t

def calculate_params(model):
    n = 0
    for init in model.graph.initializer:
        n += int(math.prod(init.dims)) if init.dims else 1
    for si in model.graph.sparse_initializer:
        n += int(math.prod(si.values.dims)) if si.values.dims else 1
    for node in model.graph.node:
        if node.op_type != 'Constant':
            continue
        for attr in node.attribute:
            if attr.name == 'value':
                n += int(math.prod(attr.t.dims)) if attr.t.dims else 1
            elif attr.name == 'value_floats':
                n += len(attr.floats)
            elif attr.name == 'value_ints':
                n += len(attr.ints)
    return n

def calculate_memory(model_path, examples, n_runs=3):
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph

    tensor_dtype = {}
    tensor_static = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not vi.type.HasField('tensor_type'):
            continue
        shape = vi.type.tensor_type.shape
        if not shape.dim:
            continue
        dims = []
        ok = True
        for d in shape.dim:
            if not d.HasField('dim_value') or d.dim_value <= 0:
                ok = False
                break
            dims.append(d.dim_value)
        if not ok:
            continue
        np_dt = onnx.helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
        tensor_dtype[vi.name] = np_dt
        tensor_static[vi.name] = int(np.prod(dims)) * np.dtype(np_dt).itemsize

    node_outputs = {n.name: list(n.output) for n in graph.node}

    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(model_path), opts, providers=['CPUExecutionProvider'])
    for p in examples[:n_runs]:
        _ = sess.run(['output'], {'input': encode_grid(p['input'])})
    trace_path = sess.end_profiling()
    with open(trace_path) as f:
        trace = json.load(f)
    os.remove(trace_path)

    tensor_runtime = {}
    for event in trace:
        if event.get('cat') != 'Node' or 'args' not in event:
            continue
        if 'output_type_shape' not in event['args']:
            continue
        node_name = event.get('name', '').replace('_kernel_time', '')
        if node_name not in node_outputs:
            continue
        outs = node_outputs[node_name]
        for i, shape_dict in enumerate(event['args']['output_type_shape']):
            if i >= len(outs):
                continue
            name = outs[i]
            if name not in tensor_dtype:
                continue
            itemsize = np.dtype(tensor_dtype[name]).itemsize
            sz = itemsize * sum(int(math.prod(dims)) for dims in shape_dict.values())
            tensor_runtime[name] = max(tensor_runtime.get(name, 0), sz)

    total = 0
    for name, static in tensor_static.items():
        if name in ('input', 'output'):
            continue
        total += max(static, tensor_runtime.get(name, 0))
    return total

def cost_and_score(model_path, examples):
    mem = calculate_memory(model_path, examples)
    model = onnx.load(str(model_path))
    params = calculate_params(model)
    cost = mem + params
    score = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return cost, score, mem, params

def verify(model_path, examples):
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    n_pass = 0
    for p in examples:
        try:
            out = sess.run(['output'], {'input': encode_grid(p['input'])})[0]
            tgt = encode_grid(p['output']) > 0.0
            if np.array_equal(out > 0.0, tgt):
                n_pass += 1
        except Exception:
            pass
    return n_pass, len(examples)

def load_examples(task_id):
    candidates = [
        COMP_DIR / f'task{task_id:03d}.json',
    ]
    for c in candidates:
        if c.exists():
            return json.load(c.open())
    for c in COMP_DIR.rglob(f'task{task_id:03d}.json'):
        return json.load(c.open())
    raise FileNotFoundError(f'task{task_id:03d}.json')

def shape_of(model, name):
    for col in (model.graph.input, model.graph.output, model.graph.value_info):
        for v in col:
            if v.name != name:
                continue
            dims = []
            for d in v.type.tensor_type.shape.dim:
                if d.HasField('dim_value'):
                    dims.append(d.dim_value)
                else:
                    return None
            return tuple(dims)
    return None

def find_reducesum_chains(model):
    nodes = list(model.graph.node)
    consumers = defaultdict(list)
    for i, n in enumerate(nodes):
        for inp in n.input:
            if inp:
                consumers[inp].append(i)
    graph_outs = {o.name for o in model.graph.output}
    chains = []
    for i, n in enumerate(nodes):
        if n.op_type != 'ReduceSum':
            continue
        out = n.output[0]
        if not out or out in graph_outs:
            continue
        cs = consumers.get(out, [])
        if len(cs) != 1:
            continue
        c_idx = cs[0]
        c_node = nodes[c_idx]
        if c_node.op_type != 'ReduceSum':
            continue
        if c_node.input and c_node.input[0] != out:
            continue
        shape = shape_of(model, out)
        size = int(np.prod(shape)) * 2 if shape else 0  # fp16 assumption for size hint
        chains.append({'first': i, 'second': c_idx, 'intermediate': out, 'shape': shape, 'size_hint': size})
    return chains

candidates_rs = []
for tid in range(1, NUM_TASKS + 1):
    src = konbu17_dir if tid in KONBU17_TASKS else afr1ste_dir
    p = src / f'task{tid:03d}.onnx'
    try:
        m = onnx.shape_inference.infer_shapes(onnx.load(str(p)), strict_mode=False)
    except Exception:
        continue
    chains = find_reducesum_chains(m)
    if chains:
        biggest = max(chains, key=lambda c: c['size_hint'])
        candidates_rs.append((tid, biggest))

candidates_rs.sort(key=lambda r: -r[1]['size_hint'])
print(f'Tasks with the ReduceSum-chain pattern: {len(candidates_rs)}')
for tid, info in candidates_rs:
    print(f'  task{tid:03d}  intermediate {info["intermediate"]:<20} shape {info["shape"]}')

def fuse_reducesum_chain(model, target_pair):
    """Replace consecutive ReduceSum nodes with a single multi-axis ReduceSum.
    target_pair is a (first_node_idx, second_node_idx) tuple as returned by the scanner.
    Returns a new model. Does not mutate the input.
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    nodes = list(m.graph.node)
    first_idx, second_idx = target_pair
    n1 = nodes[first_idx]
    n2 = nodes[second_idx]
    # In opset 13+ ReduceSum takes axes as the second input; in older opsets it is an attribute.
    # Detect whichever applies and produce a fused single-op equivalent.
    inits = {i.name: i for i in m.graph.initializer}
    def axes_for(node):
        if len(node.input) >= 2 and node.input[1] in inits:
            return list(onh.to_array(inits[node.input[1]]).flatten()), 'input'
        for attr in node.attribute:
            if attr.name == 'axes':
                return list(attr.ints), 'attr'
        return [], 'attr'
    axes_a, mode_a = axes_for(n1)
    axes_b, mode_b = axes_for(n2)
    if not axes_a or not axes_b:
        return None
    # The second ReduceSum operates on the output of the first, which has fewer rank if keepdims=0.
    # We need axes_b expressed in the original rank space. Re-index axes_b by stepping over each
    # removed axis from axes_a (assumes keepdims=0 on the first ReduceSum, which is the case in
    # the public bundle).
    def keepdims_of(node):
        for attr in node.attribute:
            if attr.name == 'keepdims':
                return attr.i
        return 1
    if keepdims_of(n1) == 0:
        removed = sorted(int(a) for a in axes_a)
        rebased = []
        for b in axes_b:
            b = int(b)
            for r in removed:
                if r <= b:
                    b += 1
            rebased.append(b)
        merged_axes = sorted(set(int(a) for a in axes_a) | set(rebased))
    else:
        merged_axes = sorted(set(int(a) for a in axes_a) | set(int(a) for a in axes_b))
    # Build the fused node: same output name as n2, input = n1.input[0], keepdims = keepdims_of(n2)
    fused_keepdims = keepdims_of(n2)
    # Always use an initializer-axes formulation if the original used initializers, otherwise
    # use an attribute. Either way the semantics are identical.
    if mode_a == 'input' or mode_b == 'input':
        axes_init_name = n2.output[0] + '_fused_axes'
        axes_init = onh.from_array(np.array(merged_axes, dtype=np.int64), name=axes_init_name)
        m.graph.initializer.append(axes_init)
        fused = oh.make_node('ReduceSum',
                              inputs=[n1.input[0], axes_init_name],
                              outputs=[n2.output[0]],
                              keepdims=fused_keepdims)
    else:
        fused = oh.make_node('ReduceSum',
                              inputs=[n1.input[0]],
                              outputs=[n2.output[0]],
                              axes=merged_axes,
                              keepdims=fused_keepdims)
    # Remove n1 and n2, insert fused at n1 position
    new_nodes = []
    for i, n in enumerate(nodes):
        if i == first_idx:
            new_nodes.append(fused)
        elif i in (first_idx, second_idx):
            continue
        else:
            new_nodes.append(n)
    # Also remove the now-skipped second
    new_nodes = [n for n in new_nodes if not (n.op_type == 'ReduceSum' and list(n.output) == list(n2.output) and n is not fused)]
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    # Drop unused initializers (e.g., the axes initializer for n1 if it has no other consumer)
    used = set()
    for n in m.graph.node:
        for inp in n.input:
            if inp:
                used.add(inp)
    new_inits = [i for i in m.graph.initializer if i.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(new_inits)
    return m

# Apply to the candidates and validate. Build a dictionary of {task_id: rewritten_bytes} for the ones that score better.
rewrites = {}
rs_receipts = []
for tid, info in candidates_rs:
    src = konbu17_dir if tid in KONBU17_TASKS else afr1ste_dir
    src_path = src / f'task{tid:03d}.onnx'
    orig = onnx.load(str(src_path))
    fused = fuse_reducesum_chain(orig, (info['first'], info['second']))
    if fused is None:
        continue
    try:
        onnx.checker.check_model(fused, full_check=True)
    except Exception as e:
        rs_receipts.append((tid, 'check_failed', str(e)[:50]))
        continue
    tmp_path = f'/tmp/task{tid:03d}_rs_fused.onnx'
    onnx.save(fused, tmp_path)
    examples = load_examples(tid)
    ex = examples['train'] + examples['test'] + examples.get('arc-gen', [])
    n_pass, n_total = verify(tmp_path, ex)
    if n_pass != n_total:
        rs_receipts.append((tid, 'verify_fail', f'{n_pass}/{n_total}'))
        os.remove(tmp_path)
        continue
    orig_cost, orig_score, _, _ = cost_and_score(src_path, ex)
    new_cost, new_score, _, _ = cost_and_score(tmp_path, ex)
    delta = new_score - orig_score
    if new_cost >= orig_cost:
        rs_receipts.append((tid, 'no_gain', f'{orig_cost} -> {new_cost}'))
        os.remove(tmp_path)
        continue
    rewrites[tid] = open(tmp_path, 'rb').read()
    rs_receipts.append((tid, 'accepted', f'{orig_cost} -> {new_cost} | score {orig_score:.3f} -> {new_score:.3f} | delta {delta:+.3f}'))
    os.remove(tmp_path)

print('ReduceSum-chain fusion receipts:')
for tid, status, info in rs_receipts:
    print(f'  task{tid:03d}: {status:<12} {info}')

def find_cast_chains(model):
    nodes = list(model.graph.node)
    consumers = defaultdict(list)
    for i, n in enumerate(nodes):
        for inp in n.input:
            if inp:
                consumers[inp].append(i)
    pairs = []
    for i, n in enumerate(nodes):
        if n.op_type != 'Cast':
            continue
        out = n.output[0]
        if not out:
            continue
        cs = consumers.get(out, [])
        if len(cs) != 1:
            continue
        c_node = nodes[cs[0]]
        if c_node.op_type != 'Cast':
            continue
        pairs.append((i, cs[0]))
    return pairs

def collapse_cast_pairs(model, pairs):
    """Remove the first Cast in each pair, route its input to the second Cast."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    nodes = list(m.graph.node)
    remove = {i for i, _ in pairs}
    remap = {nodes[i].output[0]: nodes[i].input[0] for i, _ in pairs}
    new_nodes = []
    for i, n in enumerate(nodes):
        if i in remove:
            continue
        new_inputs = [remap.get(inp, inp) for inp in n.input]
        attrs_kv = {a.name: oh.get_attribute_value(a) for a in n.attribute}
        new_nodes.append(oh.make_node(n.op_type, new_inputs, list(n.output),
                                       name=n.name if n.name else None, **attrs_kv))
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    used = set()
    for n in m.graph.node:
        for inp in n.input:
            if inp:
                used.add(inp)
    new_inits = [i for i in m.graph.initializer if i.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(new_inits)
    return m

cast_receipts = []
for tid in range(1, NUM_TASKS + 1):
    if tid in rewrites:
        continue  # already covered by Pattern 1
    src = konbu17_dir if tid in KONBU17_TASKS else afr1ste_dir
    src_path = src / f'task{tid:03d}.onnx'
    try:
        m = onnx.shape_inference.infer_shapes(onnx.load(str(src_path)), strict_mode=False)
    except Exception:
        continue
    pairs = find_cast_chains(m)
    if not pairs:
        continue
    collapsed = collapse_cast_pairs(onnx.load(str(src_path)), pairs)
    try:
        onnx.checker.check_model(collapsed, full_check=True)
    except Exception:
        continue
    tmp_path = f'/tmp/task{tid:03d}_cast_collapsed.onnx'
    onnx.save(collapsed, tmp_path)
    examples = load_examples(tid)
    ex = examples['train'] + examples['test'] + examples.get('arc-gen', [])
    n_pass, n_total = verify(tmp_path, ex)
    if n_pass != n_total:
        cast_receipts.append((tid, 'verify_fail', f'{n_pass}/{n_total}'))
        os.remove(tmp_path)
        continue
    orig_cost, orig_score, _, _ = cost_and_score(src_path, ex)
    new_cost, new_score, _, _ = cost_and_score(tmp_path, ex)
    delta = new_score - orig_score
    if new_cost >= orig_cost:
        cast_receipts.append((tid, 'no_gain', f'{orig_cost} -> {new_cost}'))
        os.remove(tmp_path)
        continue
    rewrites[tid] = open(tmp_path, 'rb').read()
    cast_receipts.append((tid, 'accepted', f'{len(pairs)} chains | cost {orig_cost} -> {new_cost} | delta {delta:+.3f}'))
    os.remove(tmp_path)

print('Cast-chain collapse receipts:')
for tid, status, info in cast_receipts:
    print(f'  task{tid:03d}: {status:<12} {info}')

from collections import Counter as _Counter
from onnx import version_converter

# Ops that natively accept u8 input across opsets 12-16.
U8_NATIVE = {'Cast', 'ReduceMax', 'Reshape', 'Slice', 'Squeeze', 'Unsqueeze',
             'Transpose', 'Gather', 'Pad', 'Concat', 'Split', 'Identity',
             'Tile', 'Expand', 'Where', 'Equal'}


def narrow_task(orig):
    # Returns (new_model, n_changes, n_targets) or (None, 0, 0).
    # 1. Dedup node names so check_model accepts the model.
    name_counts = _Counter(n.name for n in orig.graph.node if n.name)
    if any(c > 1 for c in name_counts.values()):
        seen = _Counter()
        for n in orig.graph.node:
            if n.name and name_counts[n.name] > 1:
                base = n.name
                n.name = f'{base}_d{seen[base]}'
                seen[base] += 1
        for i, n in enumerate(orig.graph.node):
            if not n.name:
                n.name = f'node_{i}'
    # 2. Upgrade opset to >=14 (ReduceMax(uint8) is opset 12+; 14 is the safe target).
    cur_opset = orig.opset_import[0].version
    target_opset = max(cur_opset, 14)
    if cur_opset < target_opset:
        try:
            model = version_converter.convert_version(orig, target_opset)
        except Exception:
            return None, 0, 0
    else:
        model = orig
    try:
        m_inf = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        return None, 0, 0
    nodes = list(model.graph.node)
    consumers = defaultdict(list)
    for i, n in enumerate(nodes):
        for inp in n.input:
            if inp:
                consumers[inp].append(i)
    bool_tensors = set()
    for vi in m_inf.graph.value_info:
        if vi.type.tensor_type.elem_type == onnx.TensorProto.BOOL:
            bool_tensors.add(vi.name)

    # 3. Find Cast(bool, fp16|fp32) targets.
    targets = []
    for i, n in enumerate(nodes):
        if n.op_type != 'Cast':
            continue
        to = next((a.i for a in n.attribute if a.name == 'to'), None)
        if to not in (1, 10):
            continue
        if not n.input or n.input[0] not in bool_tensors:
            continue
        cast_out = n.output[0]
        cons = consumers.get(cast_out, [])
        if not cons:
            continue
        targets.append((i, cast_out, cons, to))
    if not targets:
        return None, 0, 0

    # 4. Build the rewrite.
    new_cast_nodes = {}
    new_consumer_nodes = {}
    inserts_after = defaultdict(list)
    will_change_dtype = set()
    for cast_idx, cast_out, cons_idxs, orig_to in targets:
        cast_node = nodes[cast_idx]
        fp_dt = TensorProto.FLOAT if orig_to == 1 else TensorProto.FLOAT16
        new_cast_nodes[cast_idx] = oh.make_node('Cast', list(cast_node.input), list(cast_node.output),
                                                to=TensorProto.UINT8)
        will_change_dtype.add(cast_out)
        for ci in cons_idxs:
            consumer = nodes[ci]
            cons_attrs = {a.name: oh.get_attribute_value(a) for a in consumer.attribute}
            if consumer.op_type in U8_NATIVE:
                # Consumer accepts u8 directly. Its output becomes u8;
                # re-route the original output name through a Cast(u8 -> fp) bridge so downstream sees fp.
                old_out = consumer.output[0]
                new_out = old_out + '_u8'
                new_consumer_nodes[ci] = oh.make_node(consumer.op_type, list(consumer.input), [new_out],
                                                     **cons_attrs)
                inserts_after[ci].append(oh.make_node('Cast', [new_out], [old_out], to=fp_dt))
                for out in consumer.output:
                    if out:
                        will_change_dtype.add(out)
            else:
                # Consumer needs fp input. Insert Cast(u8 -> fp) bridge before it.
                bridge_out = cast_out + f'_back_{ci}'
                bridge = oh.make_node('Cast', [cast_out], [bridge_out], to=fp_dt)
                new_inputs = [bridge_out if inp == cast_out else inp for inp in consumer.input]
                new_consumer_nodes[ci] = oh.make_node(consumer.op_type, new_inputs, list(consumer.output),
                                                     **cons_attrs)
                inserts_after[cast_idx].append(bridge)

    # 5. Assemble the new graph.
    final_nodes = []
    for i, n in enumerate(nodes):
        if i in new_cast_nodes:
            final_nodes.append(new_cast_nodes[i])
        elif i in new_consumer_nodes:
            final_nodes.append(new_consumer_nodes[i])
        else:
            final_nodes.append(n)
        if i in inserts_after:
            final_nodes.extend(inserts_after[i])

    new_model = onnx.ModelProto()
    new_model.CopyFrom(model)
    del new_model.graph.node[:]
    new_model.graph.node.extend(final_nodes)
    new_vi = [vi for vi in new_model.graph.value_info if vi.name not in will_change_dtype]
    del new_model.graph.value_info[:]
    new_model.graph.value_info.extend(new_vi)
    new_model.producer_name = ''
    return new_model, len(targets), len(targets)


# Apply to every task. Keep rewrites that verify exactly and reduce local cost by >=0.005 LB.
narrow_receipts = []
for tid in range(1, NUM_TASKS + 1):
    if tid in rewrites:
        continue  # already covered by Pattern 1 or 2
    src = konbu17_dir if tid in KONBU17_TASKS else afr1ste_dir
    src_path = src / f'task{tid:03d}.onnx'
    try:
        orig_model = onnx.load(str(src_path))
    except Exception:
        continue
    result = narrow_task(orig_model)
    if result is None or result[0] is None:
        continue
    new_model, n_pat, _ = result
    try:
        onnx.checker.check_model(new_model, full_check=True)
    except Exception:
        continue
    tmp_path = f'/tmp/task{tid:03d}_narrow.onnx'
    onnx.save(new_model, tmp_path)
    examples = load_examples(tid)
    ex = examples['train'] + examples['test'] + examples.get('arc-gen', [])
    n_pass, n_total = verify(tmp_path, ex)
    if n_pass != n_total:
        narrow_receipts.append((tid, 'verify_fail', f'{n_pass}/{n_total}'))
        try: os.remove(tmp_path)
        except Exception: pass
        continue
    try:
        orig_cost, orig_score, _, _ = cost_and_score(src_path, ex)
        new_cost, new_score, _, _ = cost_and_score(tmp_path, ex)
    except Exception as exc:
        narrow_receipts.append((tid, 'cost_fail', str(exc)[:120]))
        try: os.remove(tmp_path)
        except Exception: pass
        continue
    delta = new_score - orig_score
    if new_cost is None or orig_cost is None or new_cost >= orig_cost or delta < 0.005:
        try: os.remove(tmp_path)
        except Exception: pass
        continue
    rewrites[tid] = open(tmp_path, 'rb').read()
    narrow_receipts.append((tid, 'accepted', f'{n_pat} patterns | cost {orig_cost} -> {new_cost} | delta {delta:+.3f}'))
    try: os.remove(tmp_path)
    except Exception: pass

print('Boolean-reduction dtype-narrowing receipts:')
for tid, status, info in narrow_receipts:
    print(f'  task{tid:03d}: {status:<12} {info}')

def pick_source(task_id):
    base = konbu17_dir if task_id in KONBU17_TASKS else afr1ste_dir
    return base / ('task%03d.onnx' % task_id)

with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for n in range(1, NUM_TASKS + 1):
        if n in rewrites:
            zf.writestr('task%03d.onnx' % n, rewrites[n])
        else:
            zf.write(pick_source(n), arcname='task%03d.onnx' % n)

with zipfile.ZipFile(OUTPUT_ZIP) as zf:
    names = sorted(zf.namelist())

print('Submission built')
print('  rewritten tasks  :', sorted(rewrites))
print('  total ONNX       :', len(names))
print('  zip size (bytes) :', OUTPUT_ZIP.stat().st_size)