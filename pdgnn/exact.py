"""Exact and LP reference values via PuLP + CBC.

  * OPT(G, w)     : integer optimum of weighted vertex cover  (Eq. 1)
  * OPT_LP(G, w)  : optimum of the LP relaxation              (Eq. 2)

By LP strong duality OPT_LP equals the optimal dual value (Eq. 3), so it is a
valid lower bound on OPT and the tightest certificate any feasible dual can give.
"""
from __future__ import annotations

import pulp

import numpy as np

from .instances import Instance


def _solve(inst: Instance, relax: bool, time_limit: float | None = None):
    prob = pulp.LpProblem("vertex_cover", pulp.LpMinimize)
    cat = "Continuous" if relax else "Binary"
    x = [pulp.LpVariable(f"x_{v}", lowBound=0, upBound=1, cat=cat)
         for v in range(inst.n)]
    prob += pulp.lpSum(float(inst.weights[v]) * x[v] for v in range(inst.n))
    for u, v in inst.edges:
        prob += x[int(u)] + x[int(v)] >= 1
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"CBC returned status {status!r}")
    val = float(pulp.value(prob.objective)) if inst.m else 0.0
    xv = np.array([(x[v].value() or 0.0) for v in range(inst.n)], dtype=np.float64)
    return val, xv


def solve_opt(inst: Instance, time_limit: float | None = 120.0):
    """Exact integer optimum. Returns (OPT, x_star in {0,1}^n)."""
    val, xv = _solve(inst, relax=False, time_limit=time_limit)
    return val, (xv >= 0.5).astype(np.float64)


def solve_lp(inst: Instance, time_limit: float | None = 120.0):
    """LP relaxation optimum. Returns (OPT_LP, x_frac in [0,1]^n)."""
    return _solve(inst, relax=True, time_limit=time_limit)
