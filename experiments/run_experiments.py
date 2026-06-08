"""Full experimental protocol: Tables 1-2, cross-family appendix, and the
coverage-term ablation (gamma=0 collapse vs gamma>0 active primal).

Outputs (results/):
  raw_results.json   every number, for regenerating LaTeX tables
  table1.csv         per-family, per-method ratios (primary family ER c=5)
  table2.csv         PD-GNN extrapolation to larger graphs
  per_seed.csv       per-seed learned-method metrics

Run:  python experiments/run_experiments.py            (full)
      python experiments/run_experiments.py --quick    (fast self-test)

gamma=2 (the active-primal weight) was chosen by a sweep over {1,2,3,5}: it
un-collapses the primal (mean mu 0->~0.5, ~100% of the cover from mu) while
keeping dual quality >=0.94 and giving the best/most-stable solution quality.
"""
import argparse, json, os, sys, time
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pdgnn.instances import make_dataset
from pdgnn.exact import solve_opt, solve_lp
from pdgnn.baselines import greedy_max_degree, lp_round
from pdgnn.utils import cover_cost
from pdgnn.train import TrainConfig, train_model
from pdgnn.evaluate import evaluate_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
os.makedirs(RES, exist_ok=True)

FAMILIES = [("ER", 5.0), ("ER", 3.0), ("BA", 2.0), ("BA", 3.0)]  # primary = ER c=5
PRIMARY = ("ER", 5.0)

# Learned variants: name -> (is_pdgnn, gamma).  pdgnn = active primal (the model);
# pdgnn0 = gamma=0 collapse ablation; plain = single-channel baseline.
def variants(gamma_fix):
    return [("pdgnn", True, gamma_fix), ("pdgnn0", True, 0.0), ("plain", False, gamma_fix)]


def solve_refs(insts, exact=True, tl=60.0):
    opts, optlps = [], []
    for i in insts:
        optlps.append(solve_lp(i, time_limit=tl)[0])
        opts.append(solve_opt(i, time_limit=tl)[0] if exact else np.nan)
    return np.array(opts), np.array(optlps)


def pmean(rows, key):
    v = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])], float)
    return float(np.mean(v)) if v.size else np.nan


def seed_agg(per_seed, keys):
    out = {}
    for k in keys:
        vals = np.array([p[k] for p in per_seed if np.isfinite(p.get(k, np.nan))], float)
        out[k + "_mean"] = float(np.mean(vals)) if vals.size else np.nan
        out[k + "_std"] = float(np.std(vals)) if vals.size else np.nan
    return out


def run(args):
    base = TrainConfig(max_epochs=args.epochs, patience=args.patience)
    seeds = list(range(args.seeds))
    t_start = time.time()
    out = {"config": {"seeds": seeds, "train": args.train, "val": args.val,
                      "test": args.test, "epochs": args.epochs, "T": base.T,
                      "beta": base.beta, "gamma": args.gamma, "lr": base.lr,
                      "width": base.d, "n_range": [50, 100]},
           "families": {}, "extrapolation": {}}
    saved_primary_models = []  # (seed, model) for the active-primal gamma model

    for fam, param in FAMILIES:
        tag = f"{fam}_c{param:g}" if fam == "ER" else f"{fam}_m{param:g}"
        print(f"\n===== family {tag} =====", flush=True)
        train = make_dataset(fam, param, 50, 100, args.train, base_seed=1000)
        val = make_dataset(fam, param, 50, 100, args.val, base_seed=2000)
        test = make_dataset(fam, param, 50, 100, args.test, base_seed=3000)
        t0 = time.time()
        opts, optlps = solve_refs(test, exact=True)
        print(f"  refs for {len(test)} test graphs in {time.time()-t0:.1f}s", flush=True)

        greedy_r = [cover_cost(greedy_max_degree(i), i.weights) / o for i, o in zip(test, opts)]
        lpr_cost = [cover_cost(lp_round(i), i.weights) for i in test]
        lpr_r = [c / o for c, o in zip(lpr_cost, opts)]
        lpr_cert = [c / lp for c, lp in zip(lpr_cost, optlps)]
        fam_res = {"param": param, "n_test": len(test),
                   "greedy": {"emp_mean": float(np.mean(greedy_r)), "emp_std": float(np.std(greedy_r))},
                   "lp_round": {"emp_mean": float(np.mean(lpr_r)), "emp_std": float(np.std(lpr_r)),
                                "cert_mean": float(np.mean(lpr_cert)), "cert_std": float(np.std(lpr_cert))},
                   "pdgnn": {"per_seed": []}, "pdgnn0": {"per_seed": []}, "plain": {"per_seed": []}}

        for seed in seeds:
            for name, is_pd, gamma in variants(args.gamma):
                cfg = replace(base, gamma=gamma)
                kind = "pdgnn" if is_pd else "plain"
                res = train_model(kind, train, val, cfg, seed=seed)
                rows = evaluate_model(res.model, test, opts, optlps, is_pdgnn=is_pd)
                rec = {"seed": seed, "epochs": res.epochs, "emp": pmean(rows, "emp_ratio"),
                       "mu_mean": pmean(rows, "mu_mean"), "frac_mu": pmean(rows, "frac_cover_from_mu")}
                if is_pd:
                    rec.update(cert=pmean(rows, "cert_ratio"), dual_q=pmean(rows, "dual_quality"),
                               gap=pmean(rows, "cert_gap"))
                    if name == "pdgnn" and (fam, param) == PRIMARY:
                        saved_primary_models.append((seed, res.model))
                fam_res[name]["per_seed"].append(rec)
                print(f"  seed {seed} {name:6s} g={gamma:g} {res.epochs:3d}ep emp={rec['emp']:.3f} "
                      f"%mu={100*rec['frac_mu']:.0f}"
                      + (f" cert={rec['cert']:.3f} dualQ={rec['dual_q']:.3f}" if is_pd else ""), flush=True)

        for name in ("pdgnn", "pdgnn0", "plain"):
            keys = ["emp", "frac_mu"] + (["cert", "dual_q", "gap"] if name != "plain" else [])
            fam_res[name].update(seed_agg(fam_res[name]["per_seed"], keys))
        out["families"][tag] = fam_res

    # ---- extrapolation (Table 2): the active-primal model on larger graphs ----
    fam, param = PRIMARY
    for n in args.extrap_sizes:
        big = make_dataset(fam, param, n, n, args.extrap_count, base_seed=4000 + n)
        _, optlps = solve_refs(big, exact=False)
        per_seed = []
        for seed, model in saved_primary_models:
            rows = evaluate_model(model, big, np.full(len(big), np.nan), optlps, is_pdgnn=True)
            per_seed.append({"seed": seed, "cert": pmean(rows, "cert_ratio"),
                             "dual_q": pmean(rows, "dual_quality"), "frac_mu": pmean(rows, "frac_cover_from_mu")})
        agg = seed_agg(per_seed, ["cert", "dual_q", "frac_mu"])
        out["extrapolation"][str(n)] = {"n": n, "count": len(big), "per_seed": per_seed, **agg}
        print(f"  extrap n={n}: cert={agg['cert_mean']:.3f}+/-{agg['cert_std']:.3f} "
              f"dualQ={agg['dual_q_mean']:.3f} %mu={100*agg['frac_mu_mean']:.0f}", flush=True)

    out["wall_clock_sec"] = time.time() - t_start
    with open(os.path.join(RES, "raw_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    _write_csvs(out)
    print(f"\nDONE in {out['wall_clock_sec']:.0f}s -> results/raw_results.json")


def _write_csvs(out):
    import csv
    g = out["config"]["gamma"]
    with open(os.path.join(RES, "table1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "method", "emp_mean", "emp_std", "cert_mean", "cert_std",
                    "frac_cover_from_mu", "cert_available"])
        for tag, fr in out["families"].items():
            w.writerow([tag, "Greedy", fr["greedy"]["emp_mean"], fr["greedy"]["emp_std"], "", "", "", "no"])
            w.writerow([tag, "LP+Round", fr["lp_round"]["emp_mean"], fr["lp_round"]["emp_std"],
                        fr["lp_round"]["cert_mean"], fr["lp_round"]["cert_std"], "", "yes (LP dual)"])
            w.writerow([tag, "GNN-plain", fr["plain"]["emp_mean"], fr["plain"]["emp_std"], "", "",
                        fr["plain"]["frac_mu_mean"], "no"])
            w.writerow([tag, "PD-GNN (g=0)", fr["pdgnn0"]["emp_mean"], fr["pdgnn0"]["emp_std"],
                        fr["pdgnn0"]["cert_mean"], fr["pdgnn0"]["cert_std"], fr["pdgnn0"]["frac_mu_mean"], "yes (self)"])
            w.writerow([tag, f"PD-GNN (g={g:g})", fr["pdgnn"]["emp_mean"], fr["pdgnn"]["emp_std"],
                        fr["pdgnn"]["cert_mean"], fr["pdgnn"]["cert_std"], fr["pdgnn"]["frac_mu_mean"], "yes (self)"])
    with open(os.path.join(RES, "table2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "pdgnn_cert_mean", "pdgnn_cert_std", "dual_quality_mean",
                    "dual_quality_std", "frac_cover_from_mu", "count"])
        for n, e in out["extrapolation"].items():
            w.writerow([n, e["cert_mean"], e["cert_std"], e["dual_q_mean"], e["dual_q_std"],
                        e["frac_mu_mean"], e["count"]])
    with open(os.path.join(RES, "per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "variant", "seed", "epochs", "emp_ratio", "frac_cover_from_mu",
                    "cert_ratio", "dual_quality", "cert_gap"])
        for tag, fr in out["families"].items():
            for name in ("pdgnn", "pdgnn0", "plain"):
                for p in fr[name]["per_seed"]:
                    w.writerow([tag, name, p["seed"], p["epochs"], p["emp"], p["frac_mu"],
                                p.get("cert", ""), p.get("dual_q", ""), p.get("gap", "")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--train", type=int)
    ap.add_argument("--val", type=int)
    ap.add_argument("--test", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--extrap_count", type=int)
    a = ap.parse_args()
    if a.quick:
        d = dict(seeds=2, train=40, val=15, test=20, epochs=20, patience=10, extrap_count=8)
        a.extrap_sizes = [200, 500]
    else:
        d = dict(seeds=5, train=200, val=50, test=80, epochs=100, patience=25, extrap_count=30)
        a.extrap_sizes = [200, 500, 1000]
    for k, v in d.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    run(a)
