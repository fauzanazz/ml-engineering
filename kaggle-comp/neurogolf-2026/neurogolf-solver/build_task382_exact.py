from pathlib import Path
import sys, types, os, math
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

B,C,H,W=1,10,30,30
nodes=[]; inits=[]

def init(name, arr):
    inits.append(numpy_helper.from_array(np.array(arr), name)); return name

def node(op, inputs, outputs, **attrs):
    nodes.append(helper.make_node(op, inputs, outputs, **attrs)); return outputs[0] if len(outputs)==1 else outputs

# constants
init('st0', np.array([0],np.int64)); init('st1', np.array([1],np.int64)); init('st2', np.array([2],np.int64)); init('st8', np.array([8],np.int64)); init('st9', np.array([9],np.int64)); init('st10', np.array([10],np.int64)); init('st29', np.array([29],np.int64)); init('st30', np.array([30],np.int64))
init('axes_c', np.array([1],np.int64)); init('axes_h', np.array([2],np.int64)); init('axes_w', np.array([3],np.int64))
init('starts_0', np.array([0],np.int64)); init('ends_1', np.array([1],np.int64)); init('ends_30', np.array([30],np.int64)); init('starts_29', np.array([29],np.int64)); init('ends_30a', np.array([30],np.int64))
init('zero_u8c', np.array(0,np.uint8)); init('zero_i32', np.array(0,np.int32)); init('zero_i64', np.array(0,np.int64)); init('twentynine_i32', np.array(29,np.int32)); init('twentynine_i64', np.array(29,np.int64)); init('zero_f', np.array(0,np.float32)); init('one_f', np.array(1,np.float32)); init('half_f', np.array(0.5,np.float32))
# indices shapes [1,1,30,30]
ridx=np.arange(30,dtype=np.int32).reshape(1,1,30,1).repeat(30,3)
cidx=np.arange(30,dtype=np.int32).reshape(1,1,1,30).repeat(30,2)
init('ridx', ridx); init('cidx', cidx)
init('rep_rows', np.array([1,1,30,1],np.int64)); init('rep_cols', np.array([1,1,1,30],np.int64))
init('neg_axes_h', np.array([2],np.int64)); init('neg_axes_w', np.array([3],np.int64))

# slices channel masks
node('Slice',['input','st2','st3' if False else 'st8','axes_c'],['dummy_bad']) if False else None
# channel helpers
def ch(name, start, end):
    init(f'{name}_s', np.array([start],np.int64)); init(f'{name}_e', np.array([end],np.int64))
    return node('Slice',['input',f'{name}_s',f'{name}_e','axes_c'],[name])
in_ch=[ch(f'in{i}',i,i+1) for i in range(10)]
mask2=in_ch[2]; mask8=in_ch[8]
# dynamic grid extent: padded region has no active channel; real grid is rectangular from origin.
gridsum0=node('ReduceSum',['input','axes_c'],['gridsum0'],keepdims=1)
ingrid0=node('Greater',[gridsum0,'half_f'],['ingrid0'])
ingrid0f=node('Cast',[ingrid0],['ingrid0f'],to=TensorProto.FLOAT)
row_active=node('ReduceMax',[ingrid0f],['row_active'],axes=[3],keepdims=1)
col_active=node('ReduceMax',[ingrid0f],['col_active'],axes=[2],keepdims=1)
height_f=node('ReduceSum',[row_active,'axes_h'],['height_f'],keepdims=1)
width_f=node('ReduceSum',[col_active,'axes_w'],['width_f'],keepdims=1)
height_minus_one=node('Sub',[height_f,'one_f'],['height_minus_one'])
width_minus_one=node('Sub',[width_f,'one_f'],['width_minus_one'])
bottom_idx_scalar=node('Cast',[height_minus_one],['bottom_idx_scalar'],to=TensorProto.INT32)
right_idx_scalar=node('Cast',[width_minus_one],['right_idx_scalar'],to=TensorProto.INT32)
# Build gather indices for dynamic bottom row and right col.
bottom_idx=node('Add',['ridx',bottom_idx_scalar],['bottom_idx_tmp'])
bottom_idx=node('Mul',[bottom_idx,'zero_i32'],['bottom_idx_zeroed'])
bottom_idx=node('Add',[bottom_idx,bottom_idx_scalar],['bottom_idx'])
right_idx=node('Add',['cidx',right_idx_scalar],['right_idx_tmp'])
right_idx=node('Mul',[right_idx,'zero_i32'],['right_idx_zeroed'])
right_idx=node('Add',[right_idx,right_idx_scalar],['right_idx'])
# borders, all [1,1,30,1] or [1,1,1,30]
left2=node('Slice',[mask2,'starts_0','ends_1','axes_w'],['left2'])
right2=node('GatherElements',[mask2,right_idx],['right2_full'],axis=3)
right2=node('Slice',[right2,'starts_0','ends_1','axes_w'],['right2'])
top2=node('Slice',[mask2,'starts_0','ends_1','axes_h'],['top2'])
bottom2=node('GatherElements',[mask2,bottom_idx],['bottom2_full'],axis=2)
bottom2=node('Slice',[bottom2,'starts_0','ends_1','axes_h'],['bottom2'])
top8=node('Slice',[mask8,'starts_0','ends_1','axes_h'],['top8'])
bottom8=node('GatherElements',[mask8,bottom_idx],['bottom8_full0'],axis=2)
bottom8=node('Slice',[bottom8,'starts_0','ends_1','axes_h'],['bottom8'])
left8=node('Slice',[mask8,'starts_0','ends_1','axes_w'],['left8'])
right8=node('GatherElements',[mask8,right_idx],['right8_full0'],axis=3)
right8=node('Slice',[right8,'starts_0','ends_1','axes_w'],['right8'])
# tile seeds
node('Cast',[top8],['top8_u8'],to=TensorProto.UINT8)
node('Cast',[bottom8],['bottom8_u8'],to=TensorProto.UINT8)
node('Cast',[left8],['left8_u8'],to=TensorProto.UINT8)
node('Cast',[right8],['right8_u8'],to=TensorProto.UINT8)
node('Tile',['top8_u8','rep_rows'],['top8_full'])
node('Tile',['bottom8_u8','rep_rows'],['bottom8_full'])
node('Tile',['left8_u8','rep_cols'],['left8_full'])
node('Tile',['right8_u8','rep_cols'],['right8_full'])
# cumsums. reverse via Slice steps -1? Simpler compute suffix = total - prefix before.
def prefix(mask, axis_name, out):
    node('CumSum',[mask,axis_name],[out])
    return out
prefix(left2,'axes_h','left2_pref')
prefix(right2,'axes_h','right2_pref')
prefix(top2,'axes_w','top2_pref')
prefix(bottom2,'axes_w','bottom2_pref')
# suffix count >= pos: total - prefix + current
for base,axis,cur in [('left2','axes_h','left2'),('right2','axes_h','right2'),('top2','axes_w','top2'),('bottom2','axes_w','bottom2')]:
    pref=f'{base}_pref'
    ax_name = 'axes_h' if axis=='axes_h' else 'axes_w'
    total=node('ReduceSum',[cur,ax_name],[f'{base}_total'],keepdims=1)
    tmp=node('Sub',[total,pref],[f'{base}_suf_tmp'])
    node('Add',[tmp,cur],[f'{base}_suf'])
# cast counts to int64 after tiling/broadcast
# helper generate shifted gather along width/height
def gather_width(seed_full, count, sign, name):
    # count [1,1,30,1] -> broadcast over cols
    node('Cast',[count],[name+'_cnt_i64'],to=TensorProto.INT32)
    if sign==1:
        raw=node('Sub',['cidx',name+'_cnt_i64'],[name+'_raw'])
        ge0=node('GreaterOrEqual',[raw,'zero_i32'],[name+'_ge0'])
        le29=node('LessOrEqual',[raw,'twentynine_i32'],[name+'_le29'])
    else:
        raw=node('Add',['cidx',name+'_cnt_i64'],[name+'_raw'])
        ge0=node('GreaterOrEqual',[raw,'zero_i32'],[name+'_ge0'])
        le29=node('LessOrEqual',[raw,'twentynine_i32'],[name+'_le29'])
    node('And',[ge0,le29],[name+'_valid'])
    mn=node('Max',[raw,'zero_i32'],[name+'_mn'])
    idx=node('Min',[mn,'twentynine_i32'],[name+'_idx'])
    g=node('GatherElements',[seed_full,idx],[name+'_g'],axis=3)
    gb=node('Greater',[g,'zero_u8c'],[name+'_gb'])
    return node('And',[gb,name+'_valid'],[name])
def gather_height(seed_full, count, sign, name):
    node('Cast',[count],[name+'_cnt_i64'],to=TensorProto.INT32)
    if sign==1:
        raw=node('Sub',['ridx',name+'_cnt_i64'],[name+'_raw'])
    else:
        raw=node('Add',['ridx',name+'_cnt_i64'],[name+'_raw'])
    ge0=node('GreaterOrEqual',[raw,'zero_i32'],[name+'_ge0'])
    le29=node('LessOrEqual',[raw,'twentynine_i32'],[name+'_le29'])
    node('And',[ge0,le29],[name+'_valid'])
    mn=node('Max',[raw,'zero_i32'],[name+'_mn'])
    idx=node('Min',[mn,'twentynine_i32'],[name+'_idx'])
    g=node('GatherElements',[seed_full,idx],[name+'_g'],axis=2)
    gb=node('Greater',[g,'zero_u8c'],[name+'_gb'])
    return node('And',[gb,name+'_valid'],[name])
# top: left prefix shift + means source c-count; right prefix shift - means source c+count
def any_true(mask, name):
    node('ReduceMax',[mask],[name+'_rmax'],keepdims=1)
    return node('Greater',[name+'_rmax','half_f'],[name])
present={}
for nm in ['top8','bottom8','left8','right8','left2','right2','top2','bottom2']:
    present[nm]=any_true(nm,'has_'+nm)
def gate(ray, seedp, blkp, name):
    g=node('And',[present[seedp],present[blkp]],[name+'_g'])
    return node('And',[ray,g],[name])
rays=[]
rays.append(gate(gather_width('top8_full','left2_pref',+1,'ray_tl0'),'top8','left2','ray_tl'))
rays.append(gate(gather_width('top8_full','right2_pref',-1,'ray_tr0'),'top8','right2','ray_tr'))
rays.append(gate(gather_width('bottom8_full','left2_suf',+1,'ray_bl0'),'bottom8','left2','ray_bl'))
rays.append(gate(gather_width('bottom8_full','right2_suf',-1,'ray_br0'),'bottom8','right2','ray_br'))
rays.append(gate(gather_height('left8_full','bottom2_pref',-1,'ray_lb0'),'left8','bottom2','ray_lb'))
rays.append(gate(gather_height('left8_full','top2_pref',+1,'ray_lt0'),'left8','top2','ray_lt'))
rays.append(gate(gather_height('right8_full','bottom2_suf',-1,'ray_rb0'),'right8','bottom2','ray_rb'))
rays.append(gate(gather_height('right8_full','top2_suf',+1,'ray_rt0'),'right8','top2','ray_rt'))
# OR all rays
cur=rays[0]
for i,r in enumerate(rays[1:],1): cur=node('Or',[cur,r],[f'ray_or_{i}'])
# do not overwrite 2
is2=node('Greater',[mask2,'half_f'],['is2'])
not2=node('Not',[is2],['not2'])
cur_in=node('And',[cur,ingrid0],['cur_in'])
add8b=node('And',[cur_in,not2],['add8b'])

add8f=node('Cast',[add8b],['add8f'],to=TensorProto.FLOAT)
notadd=node('Not',[add8b],['notadd'])
notaddf=node('Cast',[notadd],['notaddf'],to=TensorProto.FLOAT)
out0=node('Mul',[in_ch[0],'notaddf'],['out0'])
out8=node('Max',[in_ch[8],'add8f'],['out8'])
outs=[]
for i in range(10): outs.append(out0 if i==0 else out8 if i==8 else in_ch[i])
node('Concat',outs,['output'],axis=1)

x=helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])
y=helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,10,30,30])
model=helper.make_model(helper.make_graph(nodes,'task382_ray',[x],[y],inits),ir_version=10,opset_imports=[helper.make_opsetid('',13)])
onnx.checker.check_model(model,full_check=True)
onnx.save(model,'task382_ray.onnx')
print('saved nodes',len(nodes),'inits',len(inits),'size',Path('task382_ray.onnx').stat().st_size)
