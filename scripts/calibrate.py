"""One-seed calibration on ER(c=5): timing, mu-collapse check, ballpark metrics."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pdgnn.instances import make_dataset
from pdgnn.exact import solve_opt, solve_lp
from pdgnn.baselines import greedy_max_degree, lp_round
from pdgnn.utils import cover_cost
from pdgnn.train import TrainConfig, train_model
from pdgnn.evaluate import evaluate_model, aggregate

t0 = time.time()
train = make_dataset("ER", 5, 50, 100, count=160, base_seed=10)
val   = make_dataset("ER", 5, 50, 100, count=40,  base_seed=11)
test  = make_dataset("ER", 5, 50, 100, count=60,  base_seed=12)
print(f"datasets built in {time.time()-t0:.1f}s")

t0 = time.time()
opts  = np.array([solve_opt(i)[0] for i in test])
optlps = np.array([solve_lp(i)[0] for i in test])
print(f"solved OPT+LP for {len(test)} test graphs in {time.time()-t0:.1f}s")

# baselines on test
rg = [cover_cost(greedy_max_degree(i), i.weights)/o for i, o in zip(test, opts)]
rl = [cover_cost(lp_round(i), i.weights)/o for i, o in zip(test, opts)]
print(f"greedy  emp ratio {np.mean(rg):.3f}±{np.std(rg):.3f}")
print(f"lp+round emp ratio {np.mean(rl):.3f}±{np.std(rl):.3f}")

cfg = TrainConfig(max_epochs=120, patience=20)
for kind, is_pd in [("pdgnn", True), ("plain", False)]:
    t0 = time.time()
    res = train_model(kind, train, val, cfg, seed=0)
    dt = time.time() - t0
    rows = evaluate_model(res.model, test, opts, optlps, is_pdgnn=is_pd)
    keys = ["emp_ratio", "mu_mean", "mu_frac_ge_half"] + (
        ["cert_ratio", "dual_quality", "cert_gap"] if is_pd else [])
    agg = aggregate(rows, keys)
    print(f"\n[{kind}] trained {res.epochs} epochs (stopped), {dt:.1f}s, best_val={res.best_val:.4f}")
    print(f"   mu_mean={agg['mu_mean_mean']:.3f}  frac(mu>=.5)={agg['mu_frac_ge_half_mean']:.3f}")
    print(f"   empirical ratio = {agg['emp_ratio_mean']:.3f} ± {agg['emp_ratio_std']:.3f}")
    if is_pd:
        print(f"   certified ratio = {agg['cert_ratio_mean']:.3f} ± {agg['cert_ratio_std']:.3f}")
        print(f"   dual quality D/OPT_LP = {agg['dual_quality_mean']:.3f}")
        print(f"   certificate gap = {agg['cert_gap_mean']:.3f}")
