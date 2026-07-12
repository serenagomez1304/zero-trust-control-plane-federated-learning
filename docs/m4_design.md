# Milestone 4 — Federated Learning of the trust_eval head (design note)

M4 is the FL contribution. It trains the `trust_eval` head across K simulated
deployments that each hold a different (non-IID) slice of the synthetic trust
data, and compares aggregation strategies against a centralized reference —
the Table 2 data (paper §5, §6.2). Code: `agents/a2a/federated/`,
`evaluation/run_federated.py`, `tests/test_federated.py`.

## Setup

- **What's federated:** only the trainable head. The encoder is frozen, so a
  client update is a small flat weight vector — cheap to communicate and
  aggregate. (Locked in M2.)
- **Framework:** a hand-rolled simulation (chosen with advisor over Flower) —
  full, transparent control of all four strategies, deterministic, no heavy
  dependency / Python-3.13 install risk. The head is tiny, so it's little code.
- **Clients precompute features once** (frozen encoder) and train the head over
  the FL rounds.

## Non-IID partitioning (`partition.py`)

Dirichlet(α) over a group key (default `(label, kind)`): low α → highly skewed,
high α → near-IID. This simultaneously skews the **threat profile** (consistent
vs. rogue balance) and the **agent-population / attack mix** (legit /
cross-domain / escalation), covering two of the paper's three heterogeneity
sources; task/intent heterogeneity rides along via `kind`. α is the single dial
that feeds Table 2 and the M5 heterogeneity characterization.

## Aggregation strategies (`strategies.py`)

All operate on the head's flat weight vector.

- **FedAvg** — sample-weighted average of client weights.
- **FedProx** — FedAvg + a proximal term `(μ/2)‖w − w_global‖²` in local training.
- **SCAFFOLD** — control variates: local gradients corrected by `(c − c_i)`,
  server and client control variates updated each round (option II, server lr 1).
- **FedNova** — normalized averaging that accounts for clients taking different
  numbers of local steps `τ_i` (which happens under non-IID data sizes).

Implementations follow the canonical papers; `tests/test_federated.py` checks
each runs, learns on IID data, and that FL comes within a small gap of
centralized.

## Result (synthetic, `hashing:256` encoder, K=5)

Centralized reference ≈ 1.00. Federated matches it on IID and mild non-IID;
under strong skew (α=0.1) the strategies diverge — FedNova stays ~1.0 while
FedProx/SCAFFOLD lag (~0.86–0.87) and FedAvg holds ~0.99. This is the intended
finding: FL ≈ centralized while data stays local, and standard aggregation
strategies behave differently on this non-IID class. Numbers are a synthetic
characterization, not a real-deployment benchmark.

## Robustness (multi-seed + K sweep)

`evaluation/run_fl_robustness.py` (via `federated_train_multiseed`) reports
mean±std final accuracy over seeds, for K ∈ {5, 10, 20} × {IID, α=1.0, α=0.1}.
This replaces the single-seed point estimates with variance bands and shows the
honest picture the single run hid: accuracy falls and **variance grows** as
clients increase (less data each) and heterogeneity rises — e.g. FedAvg at α=0.1
is ~0.87±0.15 at K=5 and lower at K=20. Results:
`evaluation/results/fl_robustness_results.{csv,json}`. (The single-seed
`run_federated.py` / `federated_results.*` remains as the quick Table 2 sanity check.)

## Deferred (see TODO.md)
The real-encoder FL run; formal heterogeneity metrics correlated with FL
performance (that is M5); optional DP-SGD.
