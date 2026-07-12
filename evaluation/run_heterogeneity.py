#!/usr/bin/env python3
"""
Heterogeneity characterization sweep — §5.3 (Milestone 5)
=========================================================
Sweeps the Dirichlet concentration alpha (IID -> strongly non-IID), and for each
level measures the partition's heterogeneity (JS divergence between deployment
marginals) and the resulting FL accuracy per strategy. Reports the correlation
between heterogeneity and accuracy, and writes the results.

    python -m evaluation.run_heterogeneity
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from agents.a2a.semantic.data import generate_dataset, train_test_split
from agents.a2a.federated.heterogeneity import partition_heterogeneity, pearson
from agents.a2a.federated.partition import partition_dirichlet, partition_iid
from agents.a2a.federated.simulate import federated_train
from agents.a2a.federated.strategies import ALL_STRATEGIES

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# alpha sweep; None => IID.
ALPHAS = [None, 5.0, 1.0, 0.5, 0.1, 0.05]


def main() -> int:
    ap = argparse.ArgumentParser(description="Heterogeneity characterization sweep (§5.3)")
    ap.add_argument("--encoder", default="hashing:256")
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..seeds-1) to average")
    args = ap.parse_args()

    import numpy as np
    seeds = list(range(args.seeds))
    train, test = train_test_split(generate_dataset(n=args.n, seed=0), seed=0)

    rows = []
    print(f"[data] train={len(train)} test={len(test)} clients={args.clients} "
          f"rounds={args.rounds} seeds={seeds}\n")
    header = (f"{'alpha':>7} {'het.overall':>12} {'agent_pop':>10} {'task':>7} {'threat':>7}   "
              + " ".join(f"{s:>13}" for s in ALL_STRATEGIES))
    print(header)
    print("-" * len(header))

    for alpha in ALPHAS:
        # Heterogeneity and accuracy averaged over seeds (partition + FL seed).
        het_runs = {k: [] for k in ("overall", "agent_population", "task_distribution", "threat_profile")}
        acc_runs = {s: [] for s in ALL_STRATEGIES}
        for seed in seeds:
            shards = (partition_iid(train, args.clients, seed) if alpha is None
                      else partition_dirichlet(train, args.clients, alpha=alpha, seed=seed))
            het = partition_heterogeneity(shards)
            for k in het_runs:
                het_runs[k].append(het[k])
            for strategy in ALL_STRATEGIES:
                r = federated_train(
                    train, test, strategy,
                    encoder_spec=args.encoder, n_clients=args.clients, rounds=args.rounds,
                    alpha=(alpha if alpha is not None else 1.0), iid=(alpha is None), seed=seed,
                )
                acc_runs[strategy].append(r.final_accuracy)

        het_mean = {k: float(np.mean(v)) for k, v in het_runs.items()}
        acc_mean = {s: float(np.mean(v)) for s in ALL_STRATEGIES for v in [acc_runs[s]]}
        acc_std = {s: float(np.std(acc_runs[s])) for s in ALL_STRATEGIES}

        rows.append({
            "alpha": ("IID" if alpha is None else alpha),
            "het_overall": round(het_mean["overall"], 4),
            "het_agent_population": round(het_mean["agent_population"], 4),
            "het_task_distribution": round(het_mean["task_distribution"], 4),
            "het_threat_profile": round(het_mean["threat_profile"], 4),
            **{f"acc_{s}": round(acc_mean[s], 4) for s in ALL_STRATEGIES},
            **{f"acc_{s}_std": round(acc_std[s], 4) for s in ALL_STRATEGIES},
        })
        alpha_label = "IID" if alpha is None else f"{alpha:g}"
        print(f"{alpha_label:>7} {het_mean['overall']:>12.3f} {het_mean['agent_population']:>10.3f} "
              f"{het_mean['task_distribution']:>7.3f} {het_mean['threat_profile']:>7.3f}   "
              + " ".join(f"{acc_mean[s]:>6.3f}±{acc_std[s]:<6.3f}" for s in ALL_STRATEGIES))

    # Correlation between overall heterogeneity and (seed-mean) accuracy, per strategy.
    het_vals = [r["het_overall"] for r in rows]
    print("\nPearson correlation (overall heterogeneity vs. mean accuracy):")
    corr = {}
    for s in ALL_STRATEGIES:
        c = pearson(het_vals, [r[f"acc_{s}"] for r in rows])
        corr[s] = round(c, 3)
        print(f"  {s:9s} r = {c:+.3f}")

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    with open(os.path.join(_RESULTS_DIR, "heterogeneity_results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(_RESULTS_DIR, "heterogeneity_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"sweep": rows, "correlation": corr}, fh, indent=2)
    print(f"\n[write] {_RESULTS_DIR}/heterogeneity_results.{{csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
