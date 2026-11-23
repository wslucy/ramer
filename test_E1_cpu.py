#!/usr/bin/env python3
"""E1 CPU smoke: AST-load architecture without importing ProtT5/MolT5."""
import ast
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

p=Path(__file__).with_name("train_mechpair_light_E1_20k.py")
t=ast.parse(p.read_text(encoding="utf-8"))
names={"masked_mean","DualAxisResidueRouter","InfoNCELoss","symmetric_nce"}
nodes=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
env=dict(torch=torch,nn=nn,F=F)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),"exec"),env)
M=env["DualAxisResidueRouter"]
h=torch.randn(3,20,1024,requires_grad=True)
mask=torch.ones((3,20),dtype=torch.bool)
mask[0,-3:]=False
for mode in ("gated","uniform"):
    model=M(1024,.25,mode)
    output=model(h,mask)
    assert output["embedding"].shape==(3,1024)
    assert output["func"].shape==(3,512)
    assert output["complement"].shape==(3,512)
    assert torch.allclose(output["embedding"],env["masked_mean"](h,mask))
    assert torch.allclose(output["residue_weights"].sum(1),torch.ones(3),atol=1e-6)
    assert bool((output["residue_weights"][~mask]==0).all())
    (output["projected"].square().mean()).backward(retain_graph=True)
    assert model.route_to_embed.weight.grad is not None
gated=M(1024,.25,"gated")
optimizer=torch.optim.AdamW(gated.parameters(),lr=1e-3)
for _ in range(3):
    out=gated(h.detach(),mask)
    loss=(out["projected"]-torch.randn_like(out["projected"])).square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step()
assert gated.score[-1].weight.grad is not None
print("PASS: E1 routing/padding/512+512/gradient CPU smoke test")
