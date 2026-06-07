"""Generate paper/figures/*.pdf from raw_results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
FIG = os.path.join(REPO, "paper", "figures")
os.makedirs(FIG, exist_ok=True)


def main():
    with open(os.path.join(RES, "raw_results.json")) as f:
        R = json.load(f)
    er5 = R["families"]["ER_c5"]

    # ---- Figure 1: empirical vs certified ratios on ER c=5 ----
    methods = ["Greedy", "LP+Round", "GNN-plain", "PD-GNN"]
    emp = [er5["greedy"]["emp_mean"], er5["lp_round"]["emp_mean"],
           er5["plain"]["emp_mean"], er5["pdgnn"]["emp_mean"]]
    emp_e = [er5["greedy"]["emp_std"], er5["lp_round"]["emp_std"],
             er5["plain"]["emp_std"], er5["pdgnn"]["emp_std"]]
    cert = [np.nan, er5["lp_round"]["cert_mean"], np.nan, er5["pdgnn"]["cert_mean"]]
    cert_e = [0, er5["lp_round"]["cert_std"], 0, er5["pdgnn"]["cert_std"]]

    x = np.arange(len(methods)); width = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(x - width / 2, emp, width, yerr=emp_e, capsize=3,
           label="Empirical ratio", color="#4C72B0")
    cvals = [c if np.isfinite(c) else 0 for c in cert]
    ax.bar(x + width / 2, cvals, width, yerr=cert_e, capsize=3,
           label="Certified ratio", color="#DD8452")
    for i, c in enumerate(cert):
        if not np.isfinite(c):
            ax.text(i + width / 2, 0.03, "n/a", ha="center", va="bottom",
                    fontsize=8, rotation=90, color="gray")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("ratio to OPT  (lower is better)")
    ax.set_title("Vertex cover, $G(n,p)$, $c=5$, $n\\in[50,100]$", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max([v for v in emp + cvals]) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ratios_er5.pdf"))
    plt.close(fig)

    # ---- Figure 2: extrapolation ----
    ex = R["extrapolation"]
    ns = sorted(ex, key=lambda k: int(k))
    xn = [int(n) for n in ns]
    cert_m = [ex[n]["cert_mean"] for n in ns]
    cert_s = [ex[n]["cert_std"] for n in ns]
    dq_m = [ex[n]["dual_q_mean"] for n in ns]

    fig, ax1 = plt.subplots(figsize=(5.0, 3.6))
    ax1.errorbar(xn, cert_m, yerr=cert_s, marker="o", color="#DD8452",
                 capsize=3, label="certified ratio")
    ax1.set_xlabel("test graph size $n$")
    ax1.set_ylabel("certified ratio", color="#DD8452")
    ax1.tick_params(axis="y", labelcolor="#DD8452")
    ax1.set_xscale("log"); ax1.set_xticks(xn)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax2 = ax1.twinx()
    ax2.plot(xn, dq_m, marker="s", color="#4C72B0", label="dual quality")
    ax2.set_ylabel("dual quality $D/\\mathrm{OPT}_{LP}$", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax2.set_ylim(0, 1.05)
    ax1.axvspan(50, 100, color="green", alpha=0.07)
    ax1.text(70, ax1.get_ylim()[1] * 0.96, "train\nrange", ha="center",
             va="top", fontsize=7, color="green")
    ax1.set_title("Extrapolation beyond training size", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "extrapolation.pdf"))
    plt.close(fig)
    print(f"wrote {FIG}/ratios_er5.pdf and extrapolation.pdf")


if __name__ == "__main__":
    main()
