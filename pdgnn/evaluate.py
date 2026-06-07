"""Run a trained model on instances and compute the paper's metrics.

Per instance (Section 6.1):
  empirical_ratio  = cost(S_hat) / OPT             (OPT known)
  certified_ratio  = cost(S_hat) / D(y_hat)        (every instance, PD-GNN)
  certificate_gap  = certified_ratio - empirical_ratio   (>= 0 by Theorem 1)
  dual_quality     = D(y_hat) / OPT_LP             (1 = LP-optimal dual)
"""
from __future__ import annotations

import numpy as np
import torch

from .instances import Instance
from .model import PDGNN, build_batch
from .utils import (cover_cost, dual_feasibility_violation, dual_value,
                    is_feasible_cover, round_and_repair)


@torch.no_grad()
def infer(model, insts: list[Instance], is_pdgnn: bool):
    """Returns (mu_list, nu_list) as numpy arrays per instance."""
    b = build_batch(insts)
    out = model(b)
    mu = (out[0] if is_pdgnn else out).cpu().numpy()
    nu = out[1].cpu().numpy() if is_pdgnn else None
    mus, nus, noff, eoff = [], [], 0, 0
    for inst in insts:
        mus.append(mu[noff:noff + inst.n]); noff += inst.n
        if is_pdgnn:
            nus.append(nu[eoff:eoff + inst.m]); eoff += inst.m
        else:
            nus.append(None)
    return mus, nus


def evaluate_model(model, insts, opts, optlps, is_pdgnn: bool):
    """opts/optlps: arrays aligned with insts (opts may contain np.nan if unknown)."""
    mus, nus = infer(model, insts, is_pdgnn)
    rows = []
    for inst, mu, nu, opt, optlp in zip(insts, mus, nus, opts, optlps):
        cover = round_and_repair(mu, inst)
        assert is_feasible_cover(cover, inst)
        cost = cover_cost(cover, inst.weights)
        row = {"n": inst.n, "m": inst.m, "cost": cost,
               "opt": float(opt), "optlp": float(optlp),
               "mu_mean": float(np.mean(mu)), "mu_frac_ge_half": float(np.mean(mu >= 0.5))}
        row["emp_ratio"] = cost / opt if np.isfinite(opt) and opt > 0 else np.nan
        if is_pdgnn:
            viol = dual_feasibility_violation(nu, inst)
            assert viol < 1e-4, f"dual infeasible: {viol}"
            D = dual_value(nu)
            row["dual"] = D
            row["cert_ratio"] = cost / D if D > 1e-9 else np.inf
            row["dual_quality"] = D / optlp if optlp > 1e-9 else np.nan
            row["dual_feas_viol"] = viol
            if np.isfinite(row["emp_ratio"]):
                row["cert_gap"] = row["cert_ratio"] - row["emp_ratio"]
        rows.append(row)
    return rows


def aggregate(rows, keys):
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=float)
        out[k + "_mean"] = float(np.mean(vals)) if vals.size else np.nan
        out[k + "_std"] = float(np.std(vals)) if vals.size else np.nan
    return out
