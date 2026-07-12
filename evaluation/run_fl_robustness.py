#!/usr/bin/env python3
"""
Federated robustness sweep — variance bands over seeds and client counts (M4+)
==============================================================================
Strengthens the Table 2 result: for each client count K and heterogeneity level,
runs every strategy across several seeds and reports mean +/- std final accuracy
(plus a centralized reference). Writes results/fl_robustness_results.{csv,json}.

    python -m evaluation.run_fl_robustness
    python -m evaluation.run_fl_robustness --seeds 5 --clients 5 10 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from agents.a2a.semantic.data import generate_dataset, train_test_split
from agents.a2a.federated.simulate import centralized_accuracy, federated_train_multiseed
from agents.a2a.federated.strategies import ALL_STRATEGIES

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

LEVELS = [("IID", {"iid": True}), ("alpha=1.0", {"alpha": 1.0}), ("alpha=0.1", {"alpha": 0.1})]


def main() -> int:
    ap = argparse.ArgumentParser(description="Federated robustness sweep (variance bands)")
    ap.add_argument("--encoder", default="hashing:256")
    ap.add_argument("--clients", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..seeds-1)")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    seeds = tuple(range(args.seeds))
    train, test = train_test_split(generate_dataset(n=args.n, seed=0), seed=0)
    cen = centralized_accuracy(train, test, encoder_spec=args.encoder, seed=0)
    print(f"[data] train={len(train)} test={len(test)}  seeds={list(seeds)}  "
          f"rounds={args.rounds}  centralized_ref={cen:.3f}")

    rows = []
    for K in args.clients:
        print(f"\n=== K = {K} deployments ===")
        header = f"{'strategy':<10}" + "".join(f"{lbl:>16}" for lbl, _ in LEVELS)
        print(header)
        print("-" * len(header))
        for strategy in ALL_STRATEGIES:
            cells = []
            for label, kw in LEVELS:
                agg = federated_train_multiseed(
                    train, test, strategy, seeds=seeds,
                    encoder_spec=args.encoder, n_clients=K, rounds=args.rounds, **kw,
                )
                cells.append(f"{agg.mean_accuracy:.3f}±{agg.std_accuracy:.3f}")
                rows.append({
                    "n_clients": K, "strategy": strategy, "level": label,
                    "mean_accuracy": round(agg.mean_accuracy, 4),
                    "std_accuracy": round(agg.std_accuracy, 4),
                    "seeds": args.seeds,
                    "per_seed": [round(a, 4) for a in agg.per_seed_accuracy],
                })
            print(f"{strategy:<10}" + "".join(f"{c:>16}" for c in cells))

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]
    with open(os.path.join(_RESULTS_DIR, "fl_robustness_results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    with open(os.path.join(_RESULTS_DIR, "fl_robustness_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"centralized_ref": round(cen, 4), "results": rows}, fh, indent=2)
    print(f"\n[write] {_RESULTS_DIR}/fl_robustness_results.{{csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
