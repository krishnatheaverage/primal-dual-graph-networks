"""Non-learned baselines: Greedy max-degree deletion and LP+Round."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .exact import solve_lp
from .instances import Instance
from .utils import round_and_repair


def greedy_max_degree(inst: Instance) -> set[int]:
    """Greedy maximum-degree deletion (weight-agnostic, Section 6.1).

    Repeatedly add the current maximum-degree vertex to the cover and delete it
    and its incident edges, until no edges remain. Ties broken by lowest index.
    """
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in inst.edges:
        u, v = int(u), int(v)
        adj[u].add(v)
        adj[v].add(u)

    cover: set[int] = set()
    while True:
        best_v, best_deg = -1, 0
        for v in sorted(adj.keys()):  # lowest index wins ties
            d = len(adj[v])
            if d > best_deg:
                best_v, best_deg = v, d
        if best_deg == 0:
            break
        cover.add(best_v)
        for u in list(adj[best_v]):
            adj[u].discard(best_v)
        adj[best_v].clear()
    return cover


def lp_round(inst: Instance, x_frac: np.ndarray | None = None,
             thr: float = 0.5) -> set[int]:
    """Solve the LP relaxation, round at `thr`, and run the repair pass.

    For an exact LP solution every edge has x_u + x_v >= 1, so at least one
    endpoint is >= 1/2 and the rounded set is already a cover; the repair pass
    is a numerical safety net and matches the protocol's "same repair pass".
    """
    if x_frac is None:
        _, x_frac = solve_lp(inst)
    return round_and_repair(x_frac, inst, thr=thr)
