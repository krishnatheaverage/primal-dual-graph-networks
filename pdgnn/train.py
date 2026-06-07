"""Training loop with mini-batches of graphs and early stopping on val loss."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from .instances import Instance
from .model import PDGNN, GNNPlain, build_batch, pdgnn_loss, plain_loss


@dataclass
class TrainConfig:
    d: int = 64
    T: int = 8
    beta: float = 1.0
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 30
    min_epochs: int = 30


@dataclass
class TrainResult:
    model: torch.nn.Module
    best_val: float
    epochs: int
    history: list = field(default_factory=list)


def _val_loss(model, insts, kind, beta):
    with torch.no_grad():
        b = build_batch(insts)
        if kind == "pdgnn":
            mu, nu = model(b)
            return float(pdgnn_loss(mu, nu, b, beta))
        mu = model(b)
        return float(plain_loss(mu, b))


def train_model(kind: str, train_insts: list[Instance], val_insts: list[Instance],
                cfg: TrainConfig, seed: int) -> TrainResult:
    """kind in {'pdgnn','plain'}. Deterministic given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = (PDGNN(cfg.d, cfg.T) if kind == "pdgnn" else GNNPlain(cfg.d, cfg.T))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)

    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    idx = np.arange(len(train_insts))
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        rng.shuffle(idx)
        for s in range(0, len(idx), cfg.batch_size):
            chunk = [train_insts[i] for i in idx[s:s + cfg.batch_size]]
            b = build_batch(chunk)
            opt.zero_grad()
            if kind == "pdgnn":
                mu, nu = model(b)
                loss = pdgnn_loss(mu, nu, b, cfg.beta)
            else:
                mu = model(b)
                loss = plain_loss(mu, b)
            loss.backward()
            opt.step()
        model.eval()
        vl = _val_loss(model, val_insts, kind, cfg.beta)
        history.append(vl)
        if vl < best_val - 1e-6:
            best_val, best_state, best_epoch, wait = vl, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if epoch >= cfg.min_epochs and wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, best_val=best_val, epochs=best_epoch, history=history)
