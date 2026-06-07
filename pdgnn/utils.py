"""Shared helpers: rounding-with-repair, feasibility checks, cost/dual values."""
from __future__ import annotations

import random

import numpy as np

from .instances import Instance


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def cover_cost(cover: set[int], weights: np.ndarray) -> float:
    return float(sum(weights[v] for v in cover))


def is_feasible_cover(cover: set[int], inst: Instance) -> bool:
    for u, v in inst.edges:
        if u not in cover and v not in cover:
            return False
    return True


def round_and_repair(scores: np.ndarray, inst: Instance, thr: float = 0.5) -> set[int]:
    """Primal rounding map rho of Section 4.3.

    Threshold the scores at `thr`, then run the repair pass: while any edge is
    uncovered, add its cheaper (lower-weight) endpoint. A single deterministic
    pass over the edge list already yields a feasible cover, because once an
    edge is processed one of its endpoints is in the cover and stays there.
    """
    cover = {int(v) for v in range(inst.n) if scores[v] >= thr}
    w = inst.weights
    for u, v in inst.edges:
        u, v = int(u), int(v)
        if u not in cover and v not in cover:
            cover.add(u if w[u] <= w[v] else v)
    assert is_feasible_cover(cover, inst), "repair pass failed to produce a cover"
    return cover


def dual_value(y: np.ndarray) -> float:
    return float(np.sum(y))


def dual_feasibility_violation(y: np.ndarray, inst: Instance) -> float:
    """Max constraint violation of a dual vector y (0 means feasible).

    Dual constraints: sum_{e ~ v} y_e <= w_v for all v, and y_e >= 0.
    """
    load = np.zeros(inst.n, dtype=np.float64)
    if inst.m:
        np.add.at(load, inst.edges[:, 0], y)
        np.add.at(load, inst.edges[:, 1], y)
    viol_node = float(np.max(load - inst.weights)) if inst.n else 0.0
    viol_neg = float(np.max(-y)) if y.size else 0.0
    return max(0.0, viol_node, viol_neg)


def is_dual_feasible(y: np.ndarray, inst: Instance, tol: float = 1e-6) -> bool:
    return dual_feasibility_violation(y, inst) <= tol
