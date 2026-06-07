"""Forward/backward checks for PD-GNN: shapes, dual feasibility, training step."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from pdgnn.instances import make_er, make_ba
from pdgnn.model import PDGNN, GNNPlain, build_batch, pdgnn_loss, plain_loss
from pdgnn.utils import dual_feasibility_violation, set_seed

set_seed(0)
insts = [make_er(60, 5, 1), make_er(80, 3, 2), make_ba(70, 3, 3), make_er(10, 0.0, 4)]
b = build_batch(insts)
print(f"batch: N={b.num_nodes} M={b.num_edges} graphs={b.num_graphs}")

model = PDGNN(d=64, T=8)
mu, nu = model(b)
print(f"mu shape {tuple(mu.shape)} in [{mu.min():.3f},{mu.max():.3f}]  nu shape {tuple(nu.shape)} >=0: {bool((nu>=-1e-7).all())}")
assert mu.shape[0] == b.num_nodes and nu.shape[0] == b.num_edges
assert torch.isfinite(mu).all() and torch.isfinite(nu).all()

# per-graph dual feasibility of nu^(T) (the basis of Theorem 1)
nu_np = nu.detach().numpy()
off = 0
maxviol = 0.0
for inst in insts:
    y = nu_np[off:off + inst.m]; off += inst.m
    v = dual_feasibility_violation(y, inst)
    maxviol = max(maxviol, v)
print(f"max dual-feasibility violation across graphs: {maxviol:.2e}  (must be ~0)")
assert maxviol < 1e-5, "projected dual is infeasible!"

# loss + backward
loss = pdgnn_loss(mu, nu, b, beta=1.0)
loss.backward()
ngrad = sum(int(p.grad is not None and torch.isfinite(p.grad).all()) for p in model.parameters())
print(f"pdgnn loss {loss.item():.4f}; params with finite grad: {ngrad}/{len(list(model.parameters()))}")

# a few optimizer steps: dual value should rise (loss has -beta*dual)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
dvals = []
for _ in range(40):
    opt.zero_grad()
    mu, nu = model(b)
    l = pdgnn_loss(mu, nu, b, beta=1.0)
    l.backward(); opt.step()
    dvals.append(float(nu.sum()))
print(f"sum(nu) over 40 steps: start={dvals[0]:.3f} end={dvals[-1]:.3f} (expect increase)")

# GNN-plain
gp = GNNPlain(d=64, T=8)
mu2 = gp(b)
l2 = plain_loss(mu2, b); l2.backward()
print(f"gnn-plain mu in [{mu2.min():.3f},{mu2.max():.3f}] loss {l2.item():.4f}")
print("MODEL SMOKE PASSED")
