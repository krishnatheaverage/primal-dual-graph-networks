# Primal–Dual Graph Networks (PD-GNN)

Reference implementation and paper for **"Primal–Dual Graph Networks: Aligning
Message Passing with Approximation Algorithms for Combinatorial Optimization on
Graphs"** (Krishna Harish).

A PD-GNN is a message-passing network for NP-hard covering problems (running
example: **minimum-weight vertex cover**) that carries a coupled **primal** state
(per node) and **dual** state (per edge) over `T` rounds imitating a primal–dual
approximation algorithm. It outputs an integral cover **and** a *feasible*
solution to the dual of the LP relaxation. Because any feasible dual lower-bounds
the optimum (weak LP duality), the ratio `cost(cover) / value(dual)` is an
**always-valid, instance-specific approximation certificate** — valid for every
input regardless of whether training succeeded (Theorem 1).

The point is not better solutions; it is an **honest self-certificate** from a
learned heuristic.

## What this repo contains

```
pdgnn/                 the library
  instances.py         Erdos-Renyi G(n,p) and Barabasi-Albert generators, Uniform[0,1] weights
  exact.py             exact OPT (ILP) and OPT_LP (LP relaxation) via PuLP + CBC
  baselines.py         Greedy max-degree deletion, LP+Round
  model.py             PD-GNN and the GNN-plain ablation (PyTorch), the dual projection, losses
  train.py             mini-batch training with early stopping (Eq. 10 objective)
  evaluate.py          the paper's metrics (empirical/certified ratio, certificate gap, dual quality)
  utils.py             rounding-with-repair, feasibility checks
experiments/
  run_experiments.py   the full protocol -> results/{raw_results.json, table1.csv, table2.csv, per_seed.csv}
scripts/
  smoke_test.py        sanity checks for the non-learned pipeline
  model_smoke.py       forward/backward + dual-feasibility checks for the model
  calibrate.py         one-seed timing / behavior probe
  make_tables.py       regenerate paper/section6.tex (results section) from raw_results.json
  make_figures.py      regenerate paper/figures/*.pdf from raw_results.json
paper/
  paper.tex            the manuscript (clean LaTeX); \input{section6}
  section6.tex         the experiments section, generated from real results
  refs.bib             bibliography
results/               numbers produced by run_experiments.py
```

## Setup

```bash
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install -r requirements.txt
```

`pulp` ships the CBC MILP solver, so no separate solver install is needed.

## Reproduce

```bash
# fast sanity checks (seconds)
./.venv/bin/python scripts/smoke_test.py
./.venv/bin/python scripts/model_smoke.py

# full protocol: 4 graph families x 5 seeds + extrapolation (~20 min on a laptop CPU)
./.venv/bin/python experiments/run_experiments.py --epochs 100
#   or a fast self-test:
./.venv/bin/python experiments/run_experiments.py --quick

# regenerate the paper's results section and figures from the numbers
./.venv/bin/python scripts/make_tables.py
./.venv/bin/python scripts/make_figures.py

# build the PDF (needs a LaTeX toolchain)
cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

## What we actually found (honest summary)

Run on minimum-weight vertex cover; exact integer optima from CBC are the ground
truth. The headline numbers live in `results/` and the paper, but the qualitative
story is:

1. **The certificate is always valid, and tight.** The projected dual is feasible
   on every instance (constraint violation `~0`), so Theorem 1 holds empirically.
   The learned dual head recovers a feasible dual within a few percent of the LP
   optimum (`D(ŷ)/OPT_LP ≈ 0.98` on the training-size graphs), so the
   self-certificate is non-vacuous.
2. **The primal collapses — a finding about the objective, not the architecture.**
   The size-normalized loss (Eq. 10) rewards a large dual but puts *no* reward on
   the primal scores, so `μ → 0` and the integral cover is produced entirely by
   the **repair pass** (which is itself a Bar-Yehuda–Even-style 2-approx, hence a
   decent ~1.2 empirical ratio). PD-GNN and GNN-plain therefore get *identical*
   solution quality; the dual head is the only learned-useful output. The paper
   discusses the one-line fix (add a coverage/complementary-slackness primal term).
3. **LP+Round is the weakest method here, not the strongest.** Vertex-cover LPs are
   half-integral (Nemhauser–Trotter), so rounding all the `1/2`s up to `1` inflates
   the cover well past Greedy on these sparse graphs. This contradicts the original
   draft's placeholder ordering and is reported as-is.
4. **The certificate extrapolates.** PD-GNN trained on `n ∈ [50,100]` keeps a
   non-vacuous certificate on graphs up to `n = 1000`, in line with the
   size-independent sample-complexity result.

These are real numbers from real runs; none were tuned to match the original
draft's illustrative placeholders.

## Implementation notes (documented departures the paper leaves to the MLPs)

- The nonnegative dual increment uses **softplus**, not a hard ReLU: a hard ReLU
  suffers dead-unit collapse on a fraction of seeds (the dual gets stuck at zero
  with no gradient to recover). Feasibility and Theorem 1 are unaffected.
- Node degree enters the encoder as `log1p(deg)` (bounded; helps the extrapolation
  test stay in-distribution).
- Undirected edge/endpoint inputs are made permutation-symmetric in `(u,v)` (sums
  and absolute differences), since an undirected edge carries a single dual.

## Relation to prior work

Two neighbors also bring duality into neural algorithmic reasoning, for a
different purpose. Numeroso, Bacciu & Veličković (*Dual Algorithmic Reasoning*,
ICLR 2023) use max-flow/min-cut duality as an **auxiliary training signal**; the
learned dual has no feasibility guarantee and is never reported as a bound. He &
Vitercik (*Primal-Dual Neural Algorithmic Reasoning*, ICML 2025) align a GNN with
the primal–dual schema and use optimal supervision to **outperform** the
approximation algorithm, with an **a-priori** "can-replicate-so-inherits-the-ratio"
guarantee. PD-GNN instead constrains the dual head to be feasible by construction
and reports cost/dual as an **a-posteriori** certificate valid on every instance.

> Note: the abbreviation "PD-GNN" is close to He & Vitercik's "PDGNN." The
> contributions are distinct (a-posteriori certificate vs. a-priori
> representability), but you may wish to rename to avoid confusion.

## License

MIT (see `LICENSE`).
