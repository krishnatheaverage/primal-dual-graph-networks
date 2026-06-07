"""Full experimental protocol: Tables 1-2 plus a cross-family appendix.

Outputs (results/):
  raw_results.json   every number, for regenerating LaTeX tables
  table1.csv         per-family, per-method ratios (primary family ER c=5)
  table2.csv         PD-GNN extrapolation to larger graphs
  per_seed.csv       per-seed learned-method metrics

Run:  python experiments/run_experiments.py            (full)
      python experiments/run_experiments.py --quick    (fast self-test)
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pdgnn.instances import make_dataset
from pdgnn.exact import solve_opt, solve_lp
from pdgnn.baselines import greedy_max_degree, lp_round
from pdgnn.utils import cover_cost
from pdgnn.train import TrainConfig, train_model
from pdgnn.evaluate import evaluate_model, aggregate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
os.makedirs(RES, exist_ok=True)

FAMILIES = [("ER", 5.0), ("ER", 3.0), ("BA", 2.0), ("BA", 3.0)]  # primary = ER c=5
PRIMARY = ("ER", 5.0)


def solve_refs(insts, exact=True, tl=60.0):
    opts, optlps = [], []
    for i in insts:
        optlps.append(solve_lp(i, time_limit=tl)[0])
        opts.append(solve_opt(i, time_limit=tl)[0] if exact else np.nan)
    return np.array(opts), np.array(optlps)


def per_instance_mean(rows, key):
    v = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])], float)
    return float(np.mean(v)) if v.size else np.nan


def run(args):
    cfg = TrainConfig(max_epochs=args.epochs, patience=args.patience)
    seeds = list(range(args.seeds))
    t_start = time.time()
    out = {"config": {"seeds": seeds, "train": args.train, "val": args.val,
                      "test": args.test, "epochs": args.epochs, "T": cfg.T,
                      "beta": cfg.beta, "lr": cfg.lr, "width": cfg.d,
                      "n_range": [50, 100]}, "families": {}, "extrapolation": {}}
    saved_primary_models = []

    for fam, param in FAMILIES:
        tag = f"{fam}_c{param:g}" if fam == "ER" else f"{fam}_m{param:g}"
        print(f"\n===== family {tag} =====", flush=True)
        train = make_dataset(fam, param, 50, 100, args.train, base_seed=1000)
        val = make_dataset(fam, param, 50, 100, args.val, base_seed=2000)
        test = make_dataset(fam, param, 50, 100, args.test, base_seed=3000)
        t0 = time.time()
        opts, optlps = solve_refs(test, exact=True)
        print(f"  refs for {len(test)} test graphs in {time.time()-t0:.1f}s", flush=True)

        # deterministic baselines
        greedy_r = [cover_cost(greedy_max_degree(i), i.weights) / o
                    for i, o in zip(test, opts)]
        lpr_cost = [cover_cost(lp_round(i), i.weights) for i in test]
        lpr_r = [c / o for c, o in zip(lpr_cost, opts)]
        lpr_cert = [c / lp for c, lp in zip(lpr_cost, optlps)]  # cost / OPT_LP
        fam_res = {"param": param, "n_test": len(test),
                   "greedy": {"emp_mean": float(np.mean(greedy_r)),
                              "emp_std": float(np.std(greedy_r))},
                   "lp_round": {"emp_mean": float(np.mean(lpr_r)),
                                "emp_std": float(np.std(lpr_r)),
                                "cert_mean": float(np.mean(lpr_cert)),
                                "cert_std": float(np.std(lpr_cert))},
                   "pdgnn": {"per_seed": []}, "plain": {"per_seed": []}}

        for seed in seeds:
            for kind, is_pd in [("pdgnn", True), ("plain", False)]:
                t0 = time.time()
                res = train_model(kind, train, val, cfg, seed=seed)
                rows = evaluate_model(res.model, test, opts, optlps, is_pdgnn=is_pd)
                rec = {"seed": seed, "epochs": res.epochs,
                       "emp": per_instance_mean(rows, "emp_ratio"),
                       "mu_mean": per_instance_mean(rows, "mu_mean"),
                       "mu_frac": per_instance_mean(rows, "mu_frac_ge_half")}
                if is_pd:
                    rec.update(cert=per_instance_mean(rows, "cert_ratio"),
                               dual_q=per_instance_mean(rows, "dual_quality"),
                               gap=per_instance_mean(rows, "cert_gap"))
                    if (fam, param) == PRIMARY:
                        saved_primary_models.append((seed, res.model))
                fam_res[kind]["per_seed"].append(rec)
                print(f"  seed {seed} {kind:5s} {res.epochs:3d}ep {time.time()-t0:4.1f}s "
                      f"emp={rec['emp']:.3f}"
                      + (f" cert={rec['cert']:.3f} dualQ={rec['dual_q']:.3f}"
                         if is_pd else ""), flush=True)

        for kind in ("pdgnn", "plain"):
            ps = fam_res[kind]["per_seed"]
            for k in (["emp", "cert", "dual_q", "gap"] if kind == "pdgnn" else ["emp"]):
                vals = np.array([p[k] for p in ps if np.isfinite(p.get(k, np.nan))], float)
                fam_res[kind][k + "_mean"] = float(np.mean(vals)) if vals.size else np.nan
                fam_res[kind][k + "_std"] = float(np.std(vals)) if vals.size else np.nan
        out["families"][tag] = fam_res

    # ---- extrapolation (Table 2): primary PD-GNN models on larger graphs ----
    fam, param = PRIMARY
    for n in args.extrap_sizes:
        big = make_dataset(fam, param, n, n, args.extrap_count, base_seed=4000 + n)
        _, optlps = solve_refs(big, exact=False)  # LP only; exact is impractical
        per_seed = []
        for seed, model in saved_primary_models:
            rows = evaluate_model(model, big, np.full(len(big), np.nan), optlps, is_pdgnn=True)
            per_seed.append({"seed": seed,
                             "cert": per_instance_mean(rows, "cert_ratio"),
                             "dual_q": per_instance_mean(rows, "dual_quality")})
        cert = np.array([p["cert"] for p in per_seed], float)
        dq = np.array([p["dual_q"] for p in per_seed], float)
        out["extrapolation"][str(n)] = {
            "n": n, "count": len(big), "per_seed": per_seed,
            "cert_mean": float(np.mean(cert)), "cert_std": float(np.std(cert)),
            "dual_q_mean": float(np.mean(dq)), "dual_q_std": float(np.std(dq))}
        print(f"  extrap n={n}: cert={np.mean(cert):.3f}±{np.std(cert):.3f} "
              f"dualQ={np.mean(dq):.3f}", flush=True)

    out["wall_clock_sec"] = time.time() - t_start
    with open(os.path.join(RES, "raw_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    _write_csvs(out)
    print(f"\nDONE in {out['wall_clock_sec']:.0f}s -> results/raw_results.json")


def _write_csvs(out):
    import csv
    with open(os.path.join(RES, "table1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "method", "emp_ratio_mean", "emp_ratio_std",
                    "cert_ratio_mean", "cert_ratio_std", "cert_available"])
        for tag, fr in out["families"].items():
            w.writerow([tag, "Greedy", fr["greedy"]["emp_mean"], fr["greedy"]["emp_std"], "", "", "no"])
            w.writerow([tag, "LP+Round", fr["lp_round"]["emp_mean"], fr["lp_round"]["emp_std"],
                        fr["lp_round"]["cert_mean"], fr["lp_round"]["cert_std"], "yes (LP dual)"])
            w.writerow([tag, "GNN-plain", fr["plain"]["emp_mean"], fr["plain"]["emp_std"], "", "", "no"])
            w.writerow([tag, "PD-GNN", fr["pdgnn"]["emp_mean"], fr["pdgnn"]["emp_std"],
                        fr["pdgnn"]["cert_mean"], fr["pdgnn"]["cert_std"], "yes (self)"])
    with open(os.path.join(RES, "table2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "pdgnn_cert_ratio_mean", "pdgnn_cert_ratio_std",
                    "dual_quality_mean", "dual_quality_std", "count"])
        for n, e in out["extrapolation"].items():
            w.writerow([n, e["cert_mean"], e["cert_std"], e["dual_q_mean"],
                        e["dual_q_std"], e["count"]])
    with open(os.path.join(RES, "per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "method", "seed", "epochs", "emp_ratio",
                    "cert_ratio", "dual_quality", "cert_gap", "mu_mean"])
        for tag, fr in out["families"].items():
            for kind in ("pdgnn", "plain"):
                for p in fr[kind]["per_seed"]:
                    w.writerow([tag, kind, p["seed"], p["epochs"], p["emp"],
                                p.get("cert", ""), p.get("dual_q", ""),
                                p.get("gap", ""), p["mu_mean"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
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
        d = dict(seeds=5, train=200, val=50, test=80, epochs=150, patience=25, extrap_count=30)
        a.extrap_sizes = [200, 500, 1000]
    for k, v in d.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    run(a)
