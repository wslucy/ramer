#!/usr/bin/env python3
"""不加载 ProtT5/MolT5 的 CPU 单元测试，检验张量公式和 B 采样器。"""
import ast
import math
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent

def load_objects(path, names):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    env={'torch':torch,'nn':nn,'F':F,'random':random}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env

legacy=load_objects(ROOT/'train_mechpair_light_MLP_20k.py',
        ['ReactionGuidedSelector','ReactionQKVSelector','MultiSlotFusion','floor_diversity'])
for cls in ('ReactionGuidedSelector','ReactionQKVSelector'):
    module=legacy[cls](32,topk=5,**({'heads':4} if cls=='ReactionQKVSelector' else {}))
    res=torch.randn(2,19,32,requires_grad=True)
    rxn=torch.randn(2,32,requires_grad=True)
    mask=torch.ones((2,19),dtype=torch.bool)
    mask[0,-4:]=False
    local,inds,att=module(res,rxn,mask)
    assert tuple(local.shape)==(2,32)
    assert tuple(inds.shape)==(2,5)
    assert tuple(att.shape)==(2,19)
    assert torch.allclose(att.masked_select(~mask),torch.zeros_like(att.masked_select(~mask)))
    assert torch.allclose(att.sum(1),torch.ones(2),atol=1e-5)
    local.square().sum().backward()
    assert res.grad.abs().sum()>0 and rxn.grad.abs().sum()>0

fus=legacy['MultiSlotFusion'](32,16,4,16,4)
s=torch.randn(2,18,32,requires_grad=True)
g=torch.randn(2,15,32,requires_grad=True)
sm=torch.ones(2,18,dtype=torch.bool)
gm=torch.ones(2,15,dtype=torch.bool)
sm[0,-5:]=False
gm[0,-4:]=False
fused,div=fus(s,sm,g,gm)
assert fused.shape==(2,16)
assert 0 <= div.item() <= 1.01
loss=(fused**2).mean()+0.01*legacy['floor_diversity'](div,.1,5.)
loss.backward()
assert s.grad.abs().sum()>0 and g.grad.abs().sum()>0

for val,sgn in [(.05,-1),(.15,1)]:
    raw=torch.tensor(val,requires_grad=True)
    eff=legacy['floor_diversity'](raw,.1,5.)
    eff.backward()
    assert (raw.grad.item()>0)==(sgn>0)
assert legacy['floor_diversity'](torch.tensor(.1),.1,5).item()==0

mod=load_objects(ROOT/'train_mechpair_AB_pair_sampler.py',['RheaPairBatchSampler'])
records=[]
for gid in range(40):
    for j in range(20):
        records.append({'RHEA':f'R{gid}', 'sequence':f'S{gid}_{j}', 'primaryAccession':f'P{gid}_{j}'})
ds=SimpleNamespace(records=records)
for rank in range(4):
    smp=mod['RheaPairBatchSampler'](ds,list(range(len(records))),16,rank,4,pairs_per_batch=2)
    smp.set_epoch(0)
    batches=list(iter(smp))
    assert len(batches)==len(smp)==12
    for b in batches:
        assert len(b)==16 and len(set(b))==16
        count=sum(records[b[i]]['RHEA']==records[b[j]]['RHEA']
                  for i in range(16) for j in range(i+1,16))
        assert count>=2
    smp.set_epoch(0)
    assert list(iter(smp))==batches
print('CPU TESTS PASS: MLP, QKV, fusion, Floor gradients, 4-rank B sampler')
