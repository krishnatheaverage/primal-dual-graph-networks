"""Sanity checks for the non-learned pipeline (no torch needed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pdgnn.instances import make_er, make_ba, make_dataset
from pdgnn.exact import solve_opt, solve_lp
from pdgnn.baselines import greedy_max_degree, lp_round
from pdgnn.utils import cover_cost, is_feasible_cover, is_dual_feasible

def check(inst, label):
    opt, xopt = solve_opt(inst)
    optlp, xfrac = solve_lp(inst)
    g = greedy_max_degree(inst)
    lr = lp_round(inst, xfrac)
    cg, cl = cover_cost(g, inst.weights), cover_cost(lr, inst.weights)
    assert is_feasible_cover(g, inst), "greedy infeasible"
    assert is_feasible_cover(lr, inst), "lp_round infeasible"
    # ordering: OPT_LP <= OPT <= cost(any feasible cover)
    eps = 1e-6
    assert optlp <= opt + eps, f"OPT_LP {optlp} > OPT {opt}"
    assert opt <= cg + eps and opt <= cl + eps, "OPT exceeds a feasible cover cost"
    print(f"[{label}] n={inst.n} m={inst.m} avgdeg={inst.avg_degree:.2f} "
          f"OPT_LP={optlp:.3f} OPT={opt:.3f} greedy={cg:.3f}({cg/opt:.3f}) "
          f"lp_round={cl:.3f}({cl/opt:.3f})")
    return opt, optlp, cg, cl

print("== single-instance checks ==")
check(make_er(60, 5, seed=1), "ER c=5")
check(make_er(80, 3, seed=2), "ER c=3")
check(make_ba(60, 2, seed=3), "BA m=2")
check(make_ba(90, 3, seed=4), "BA m=3")

# empty-graph edge case
empty = make_er(10, 0.0, seed=5)
opt, _ = solve_opt(empty)
print(f"== empty graph: m={empty.m} OPT={opt} (expect 0) ==")
assert opt == 0.0

# dual feasibility helper: all-zero dual is feasible; huge dual is not
inst = make_er(40, 5, seed=6)
assert is_dual_feasible(np.zeros(inst.m), inst)
assert not is_dual_feasible(np.full(inst.m, 10.0), inst)

# small aggregate to eyeball average ratios on the Table-1 setting (ER c=5)
print("\n== aggregate on 25 ER(c=5) graphs, n in [50,100] ==")
ds = make_dataset("ER", 5, 50, 100, count=25, base_seed=123)
rg, rl = [], []
for inst in ds:
    opt, xopt = solve_opt(inst); _, xf = solve_lp(inst)
    rg.append(cover_cost(greedy_max_degree(inst), inst.weights) / opt)
    rl.append(cover_cost(lp_round(inst, xf), inst.weights) / opt)
print(f"greedy   empirical ratio: mean={np.mean(rg):.3f} std={np.std(rg):.3f}")
print(f"lp_round empirical ratio: mean={np.mean(rl):.3f} std={np.std(rl):.3f}")
print("\nALL CHECKS PASSED")
