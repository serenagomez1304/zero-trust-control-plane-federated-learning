#!/usr/bin/env python3
"""
Per-hop overhead & ablation — §6.3 (Milestone 6)
================================================
Reports the per-stage latency breakdown, total per-hop overhead, and throughput
for three configurations, so the trust model's cost can be compared against the
principal-based baseline:

  - stub      — the M1 stub semantic model (floor)
  - baseline  — principal-based trust (identity check, no purpose-scoped view)
  - ours      — the learned per-message model

    python -m evaluation.run_overhead
    python -m evaluation.run_overhead --encoder sentence-transformers:all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os

from agents.a2a.trust import StubSemanticModel
from evaluation.attacks import PrincipalBaselineModel
from evaluation.overhead import STAGES, measure_overhead

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _build_ours(encoder: str):
    from agents.a2a.semantic.data import generate_dataset, train_test_split
    from agents.a2a.semantic.encoders import get_encoder
    from agents.a2a.semantic.model import RealSemanticModel, TrustEvalModel

    train, _ = train_test_split(generate_dataset(n=900, seed=0), seed=0)
    model = TrustEvalModel(get_encoder(encoder))
    model.fit(train, epochs=30, seed=0)
    return RealSemanticModel(model)


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-hop overhead & ablation (§6.3)")
    ap.add_argument("--encoder", default="hashing:256",
                    help="encoder for 'ours' (e.g. sentence-transformers:all-MiniLM-L6-v2)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    print(f"[setup] building 'ours' model (encoder={args.encoder})")
    configs = {
        "stub": StubSemanticModel(),
        "baseline": PrincipalBaselineModel(),
        "ours": _build_ours(args.encoder),
    }

    results = []
    for name, model in configs.items():
        results.append(measure_overhead(name, model, iters=args.iters, warmup=args.warmup))

    # Per-stage table
    print(f"\nPer-hop latency by stage (ms, mean)   iters={args.iters}\n")
    header = f"{'config':<10}" + "".join(f"{s:>12}" for s in STAGES) + f"{'TOTAL':>12}{'hops/s':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r.config:<10}" + "".join(f"{r.stages[s].mean_ms:>12.4f}" for s in STAGES)
        line += f"{r.total_ms:>12.4f}{r.throughput_hops_per_s:>10.0f}"
        print(line)

    ours = next(r for r in results if r.config == "ours")
    base = next(r for r in results if r.config == "baseline")
    print(f"\nper-hop overhead of ours vs. baseline: "
          f"{ours.total_ms - base.total_ms:+.4f} ms  "
          f"(trust_eval is {ours.stages['trust_eval'].mean_ms:.4f} ms of it)")

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    rows = []
    for r in results:
        row = {"config": r.config, "total_ms": round(r.total_ms, 5),
               "throughput_hops_per_s": round(r.throughput_hops_per_s, 1)}
        for s in STAGES:
            row[f"{s}_ms"] = round(r.stages[s].mean_ms, 5)
        rows.append(row)
    with open(os.path.join(_RESULTS_DIR, "overhead_results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(_RESULTS_DIR, "overhead_results.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n[write] {_RESULTS_DIR}/overhead_results.{{csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
