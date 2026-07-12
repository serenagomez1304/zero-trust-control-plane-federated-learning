#!/usr/bin/env python3
"""
Federated training sweep — Table 2 (Milestone 4)
================================================
Trains the trust head with each aggregation strategy at several heterogeneity
levels, alongside a centralized reference, and writes the results table.

    python -m evaluation.run_federated
    python -m evaluation.run_federated --clients 10 --rounds 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from agents.a2a.semantic.data import generate_dataset, train_test_split
from agents.a2a.federated.simulate import centralized_accuracy, federated_train
from agents.a2a.federated.strategies import ALL_STRATEGIES

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Heterogeneity levels: label -> federated_train kwargs.
LEVELS = [
    ("IID", {"iid": True}),
    ("alpha=1.0", {"alpha": 1.0}),
    ("alpha=0.1", {"alpha": 0.1}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Federated training sweep (Table 2)")
    ap.add_argument("--encoder", default="hashing:256")
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--n", type=int, default=2000, help="synthetic dataset size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, test = train_test_split(generate_dataset(n=args.n, seed=args.seed), seed=args.seed)
    print(f"[data] train={len(train)} test={len(test)}  clients={args.clients} rounds={args.rounds}")

    cen = centralized_accuracy(train, test, encoder_spec=args.encoder, seed=args.seed)
    print(f"[ref]  centralized accuracy = {cen:.3f}\n")

    rows = []
    for strategy in ALL_STRATEGIES:
        cells = []
        for label, kw in LEVELS:
            r = federated_train(
                train, test, strategy,
                encoder_spec=args.encoder, n_clients=args.clients, rounds=args.rounds,
                seed=args.seed, **kw,
            )
            rows.append({
                "strategy": strategy, "level": label,
                "final_accuracy": round(r.final_accuracy, 4),
                "rounds_to_converge": r.rounds_to_converge,
                "n_clients": args.clients, "alpha": r.alpha,
            })
            cells.append(f"{label}={r.final_accuracy:.3f}")
        print(f"  {strategy:9s}  " + "  ".join(cells))

    # Table 2 (final accuracy)
    print("\nTable 2 — Federated accuracy by strategy and heterogeneity "
          f"(centralized ref = {cen:.3f})")
    header = f"{'strategy':<10}" + "".join(f"{lbl:>12}" for lbl, _ in LEVELS)
    print(header)
    print("-" * len(header))
    for strategy in ALL_STRATEGIES:
        line = f"{strategy:<10}"
        for label, _ in LEVELS:
            acc = next(r["final_accuracy"] for r in rows if r["strategy"] == strategy and r["level"] == label)
            line += f"{acc:>12.3f}"
        print(line)

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    rows.append({"strategy": "centralized", "level": "pooled", "final_accuracy": round(cen, 4),
                 "rounds_to_converge": None, "n_clients": 1, "alpha": None})
    with open(os.path.join(_RESULTS_DIR, "federated_results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(_RESULTS_DIR, "federated_results.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n[write] {_RESULTS_DIR}/federated_results.{{csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
