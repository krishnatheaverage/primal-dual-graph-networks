"""PD-GNN and the GNN-plain ablation (Section 4), in plain PyTorch.

A batch is a disjoint union of graphs so a whole training mini-batch is one
forward pass. Node/edge -> graph index maps recover per-graph sums for the
1/n-normalized loss of Eq. 10.

Faithful to Section 4 with two documented implementation choices the text
leaves to the MLPs:
  * degree enters the node encoder as log1p(deg) (bounded; helps the
    extrapolation test of Section 6 stay in-distribution);
  * edge/endpoint inputs are made permutation-symmetric in (u, v) by feeding
    sums and absolute differences, since an undirected edge carries one dual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .instances import Instance


# --------------------------------------------------------------------------- #
# Batched disjoint-union representation
# --------------------------------------------------------------------------- #
@dataclass
class Batch:
    w: torch.Tensor            # [N]    node weights
    deg: torch.Tensor          # [N]    node degrees (raw)
    edge_index: torch.Tensor   # [2, M] global endpoint ids (u, v), u < v
    node_graph: torch.Tensor   # [N]    graph id per node
    edge_graph: torch.Tensor   # [M]    graph id per edge
    n_per_graph: torch.Tensor  # [B]    node count per graph
    num_nodes: int
    num_edges: int
    num_graphs: int

    def to(self, device):
        for f in ("w", "deg", "edge_index", "node_graph", "edge_graph", "n_per_graph"):
            setattr(self, f, getattr(self, f).to(device))
        return self


def build_batch(insts: list[Instance], device="cpu") -> Batch:
    ws, degs, eidx, ng, eg, npg = [], [], [], [], [], []
    node_off = 0
    for gi, inst in enumerate(insts):
        ws.append(inst.weights.astype(np.float32))
        degs.append(inst.degrees.astype(np.float32))
        if inst.m:
            e = inst.edges.T.astype(np.int64) + node_off  # [2, m]
            eidx.append(e)
            eg.append(np.full(inst.m, gi, dtype=np.int64))
        ng.append(np.full(inst.n, gi, dtype=np.int64))
        npg.append(inst.n)
        node_off += inst.n
    w = torch.from_numpy(np.concatenate(ws))
    deg = torch.from_numpy(np.concatenate(degs))
    edge_index = (torch.from_numpy(np.concatenate(eidx, axis=1))
                  if eidx else torch.zeros((2, 0), dtype=torch.long))
    node_graph = torch.from_numpy(np.concatenate(ng))
    edge_graph = (torch.from_numpy(np.concatenate(eg))
                  if eg else torch.zeros((0,), dtype=torch.long))
    n_per_graph = torch.tensor([float(x) for x in npg], dtype=torch.float32)
    b = Batch(w, deg, edge_index, node_graph, edge_graph, n_per_graph,
              int(w.numel()), int(edge_index.shape[1]), len(insts))
    return b.to(device)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Two hidden layers of width `hidden`, ReLU (Section 6.1)."""

    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _scatter_add(src, index, n):
    out = src.new_zeros((n,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def project_dual(nu_tilde, edge_index, w, num_nodes):
    """Dual feasibility projection Pi_Y of Eq. (8).

    load_v = sum_{e~v} nu_tilde_e ; lam_v = min(1, w_v / load_v) ;
    nu_e   = min(lam_u, lam_v) * nu_tilde_e   in Y(G, w).
    """
    if nu_tilde.numel() == 0:
        return nu_tilde
    u, v = edge_index[0], edge_index[1]
    load = _scatter_add(nu_tilde, u, num_nodes) + _scatter_add(nu_tilde, v, num_nodes)
    safe = load.clamp_min(1e-12)
    lam = torch.minimum(torch.ones_like(load), w / safe)
    lam = torch.where(load > 0, lam, torch.ones_like(load))
    factor = torch.minimum(lam[u], lam[v])
    return factor * nu_tilde


# --------------------------------------------------------------------------- #
# PD-GNN
# --------------------------------------------------------------------------- #
class PDGNN(nn.Module):
    def __init__(self, d=64, T=8):
        super().__init__()
        self.d, self.T = d, T
        self.phi_enc = MLP(2, d)                 # [w_v, log1p deg] -> h
        self.psi_enc = MLP(2, d)                 # [min(w_u,w_v), max(..)] -> g
        self.m_dual = MLP(3 * d, 1)              # [g, h_u+h_v, |h_u-h_v|] -> delta
        self.u_dual = MLP(3 * d + 1, d)          # + nu_e -> g'
        self.m_prim = MLP(d + 1, d)              # [h_nbr, nu_e] -> message
        self.u_prim = MLP(2 * d + 1, d)          # [h_v, sigma_v, agg] -> h'
        self.readout = nn.Linear(d, 1)           # a^T h + b -> logit

    def forward(self, b: Batch):
        u, v = b.edge_index[0], b.edge_index[1]
        N, M = b.num_nodes, b.num_edges
        h = self.phi_enc(torch.stack([b.w, torch.log1p(b.deg)], dim=1))
        if M:
            wu, wv = b.w[u], b.w[v]
            g = self.psi_enc(torch.stack([torch.minimum(wu, wv),
                                          torch.maximum(wu, wv)], dim=1))
            nu = torch.zeros(M, device=b.w.device)
        else:
            g = torch.zeros((0, self.d), device=b.w.device)
            nu = torch.zeros(0, device=b.w.device)

        for _ in range(self.T):
            if M:
                # (i) dual update + feasibility projection (Eqs. 6, 8).
                # softplus (not a hard ReLU) for the nonnegative increment: a hard
                # ReLU suffers dead-unit collapse on some seeds (delta stuck at 0,
                # no gradient to recover the dual), which softplus avoids. delta >= 0
                # and feasibility/Theorem 1 are unchanged.
                sym_sum = h[u] + h[v]
                sym_dif = (h[u] - h[v]).abs()
                delta = F.softplus(self.m_dual(torch.cat([g, sym_sum, sym_dif], 1))).squeeze(-1)
                nu = project_dual(nu + delta, b.edge_index, b.w, N)
                g = self.u_dual(torch.cat([g, nu.unsqueeze(-1), sym_sum, sym_dif], 1))
                # (ii) slack-coupled primal update (Eq. 7)
                s = _scatter_add(nu, u, N) + _scatter_add(nu, v, N)
                sigma = torch.relu(b.w - s)
                msg_to_u = self.m_prim(torch.cat([h[v], nu.unsqueeze(-1)], 1))
                msg_to_v = self.m_prim(torch.cat([h[u], nu.unsqueeze(-1)], 1))
                agg = _scatter_add(msg_to_u, u, N) + _scatter_add(msg_to_v, v, N)
            else:
                sigma = b.w
                agg = torch.zeros((N, self.d), device=b.w.device)
            h = self.u_prim(torch.cat([h, sigma.unsqueeze(-1), agg], 1))

        mu = torch.sigmoid(self.readout(h).squeeze(-1))   # [N] in [0,1]
        return mu, nu                                     # nu = nu^(T) in Y


# --------------------------------------------------------------------------- #
# GNN-plain: single primal channel, no dual head, no certificate
# --------------------------------------------------------------------------- #
class GNNPlain(nn.Module):
    def __init__(self, d=64, T=8):
        super().__init__()
        self.d, self.T = d, T
        self.phi_enc = MLP(2, d)
        self.m_prim = MLP(d, d)
        self.u_prim = MLP(2 * d, d)
        self.readout = nn.Linear(d, 1)

    def forward(self, b: Batch):
        u, v = b.edge_index[0], b.edge_index[1]
        N, M = b.num_nodes, b.num_edges
        h = self.phi_enc(torch.stack([b.w, torch.log1p(b.deg)], dim=1))
        for _ in range(self.T):
            if M:
                agg = (_scatter_add(self.m_prim(h[v]), u, N)
                       + _scatter_add(self.m_prim(h[u]), v, N))
            else:
                agg = torch.zeros((N, self.d), device=b.w.device)
            h = self.u_prim(torch.cat([h, agg], 1))
        return torch.sigmoid(self.readout(h).squeeze(-1))


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def pdgnn_loss(mu, nu, b: Batch, beta=1.0):
    """Eq. 10: mean over graphs of (1/n) sum_v w_v mu_v - (beta/n) sum_e y_e."""
    prim = _scatter_add(b.w * mu, b.node_graph, b.num_graphs)
    dual = (_scatter_add(nu, b.edge_graph, b.num_graphs) if b.num_edges
            else torch.zeros(b.num_graphs, device=b.w.device))
    per_graph = (prim - beta * dual) / b.n_per_graph
    return per_graph.mean()


def plain_loss(mu, b: Batch):
    """Soft primal cost alone (the GNN-plain objective)."""
    prim = _scatter_add(b.w * mu, b.node_graph, b.num_graphs)
    return (prim / b.n_per_graph).mean()
