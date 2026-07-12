#!/usr/bin/env python3
"""
Run the rogue-agent attack harness and emit Table 1 (Milestone 3)
=================================================================
Runs all three attack classes through the principal-based baseline and our
per-message model, prints the results table, and writes CSV + JSON to
``evaluation/results/``.

By default it loads the trained trust_eval artifact (real encoder). If that is
unavailable it trains a quick, offline hashing-encoder model so the harness
always runs.

    python -m evaluation.run_attacks
    python -m evaluation.run_attacks --encoder hashing:256   # force offline model
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os

from agents.a2a.trust import DEFAULT_ARTIFACT
from evaluation.attacks import PrincipalBaselineModel, evaluate, format_table

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load_ours(encoder: str | None):
    """Load the trained model, or fall back to a freshly trained offline model."""
    from agents.a2a.semantic.model import RealSemanticModel, TrustEvalModel

    if encoder is None and os.path.exists(DEFAULT_ARTIFACT):
        print(f"[model]  loading trained artifact {DEFAULT_ARTIFACT}")
        return RealSemanticModel(TrustEvalModel.load(DEFAULT_ARTIFACT))

    spec = encoder or "hashing:256"
    print(f"[model]  no artifact / override given — training offline model ({spec})")
    from agents.a2a.semantic.data import generate_dataset, train_test_split
    from agents.a2a.semantic.encoders import get_encoder

    train, _ = train_test_split(generate_dataset(n=1600, seed=0), seed=0)
    model = TrustEvalModel(get_encoder(spec))
    model.fit(train, epochs=40, seed=0)
    return RealSemanticModel(model)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the rogue-agent attack harness")
    ap.add_argument("--encoder", default=None,
                    help="force this encoder spec instead of the trained artifact (e.g. 'hashing:256')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    ours = _load_ours(args.encoder)
    metrics = evaluate(ours, PrincipalBaselineModel(), seed=args.seed, threshold=args.threshold)

    print()
    print(format_table(metrics))
    print()

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    rows = [dataclasses.asdict(m) for m in metrics]

    csv_path = os.path.join(_RESULTS_DIR, "attack_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(_RESULTS_DIR, "attack_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    print(f"[write]  {csv_path}")
    print(f"[write]  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
