"""Generate paper/figures/*.pdf from raw_results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "results", "raw_results.json")
FIG = os.path.join(REPO, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
FAM_ORDER = ["ER_c5", "ER_c3", "BA_m2", "BA_m3"]
FAM_LBL = {"ER_c5": "ER c=5", "ER_c3": "ER c=3", "BA_m2": "BA m=2", "BA_m3": "BA m=3"}


def main():
    R = json.load(open(RAW))
    fams = R["families"]
    g = R["config"]["gamma"]

    # ---- Figure 1: per-family ablation of empirical ratio ----
    labels = [FAM_LBL[t] for t in FAM_ORDER]
    lpr = [fams[t]["lp_round"]["emp_mean"] for t in FAM_ORDER]
    g0 = [fams[t]["pdgnn0"]["emp_mean"] for t in FAM_ORDER]
    g0e = [fams[t]["pdgnn0"]["emp_std"] for t in FAM_ORDER]
    gx = [fams[t]["pdgnn"]["emp_mean"] for t in FAM_ORDER]
    gxe = [fams[t]["pdgnn"]["emp_std"] for t in FAM_ORDER]

    x = np.arange(len(FAM_ORDER)); w = 0.27
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(x - w, lpr, w, label="LP+Round", color="#999999")
    ax.bar(x, g0, w, yerr=g0e, capsize=2, label=r"PD-GNN $\gamma{=}0$ (repair)", color="#4C72B0")
    ax.bar(x + w, gx, w, yerr=gxe, capsize=2, label=rf"PD-GNN $\gamma{{=}}{g:g}$ (learned)", color="#DD8452")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("empirical ratio to OPT  (lower better)")
    ax.set_title("Coverage term: learned primal vs. repair", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0.9, max(lpr + g0 + gx) * 1.12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ablation.pdf")); plt.close(fig)

    # ---- Figure 2: extrapolation ----
    ex = R["extrapolation"]
    ns = sorted(ex, key=lambda k: int(k)); xn = [int(n) for n in ns]
    cert_m = [ex[n]["cert_mean"] for n in ns]; cert_s = [ex[n]["cert_std"] for n in ns]
    dq_m = [ex[n]["dual_q_mean"] for n in ns]

    fig, ax1 = plt.subplots(figsize=(5.0, 3.6))
    ax1.errorbar(xn, cert_m, yerr=cert_s, marker="o", color="#DD8452", capsize=3)
    ax1.set_xlabel("test graph size $n$")
    ax1.set_ylabel("certified ratio", color="#DD8452")
    ax1.tick_params(axis="y", labelcolor="#DD8452")
    ax1.set_xscale("log"); ax1.set_xticks(xn)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax2 = ax1.twinx()
    ax2.plot(xn, dq_m, marker="s", color="#4C72B0")
    ax2.set_ylabel(r"dual quality $D/\mathrm{OPT}_{LP}$", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0"); ax2.set_ylim(0, 1.05)
    ax1.set_title(rf"Extrapolation ($\gamma{{=}}{g:g}$ model)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "extrapolation.pdf")); plt.close(fig)
    print(f"wrote {FIG}/ablation.pdf and extrapolation.pdf")


if __name__ == "__main__":
    main()
