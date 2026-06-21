#!/usr/bin/env python3
"""
Train the trust_eval head
=========================
Generate synthetic data, train the trainable head on top of a frozen encoder,
report metrics, and save an artifact that ``resolve_mu`` can load.

Examples:
    # Default: real sentence-transformer encoder -> default artifact path
    python -m agents.a2a.semantic.train

    # Offline / reproducible: deterministic hashing encoder
    python -m agents.a2a.semantic.train --encoder hashing:256 --out /tmp/trust_eval_hashing.pt
"""

from __future__ import annotations

import argparse
import os

from agents.a2a.semantic.data import generate_dataset, train_test_split
from agents.a2a.semantic.encoders import get_encoder
from agents.a2a.semantic.model import TrustEvalModel

DEFAULT_ARTIFACT = os.path.join(os.path.dirname(__file__), "artifacts", "trust_eval_v0.pt")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the trust_eval head")
    ap.add_argument("--encoder", default="sentence-transformers:all-MiniLM-L6-v2",
                    help="encoder spec (e.g. 'hashing:256' or 'sentence-transformers:all-MiniLM-L6-v2')")
    ap.add_argument("--n", type=int, default=1600, help="number of synthetic examples")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_ARTIFACT, help="artifact output path")
    args = ap.parse_args()

    print(f"[data]    generating {args.n} synthetic examples (seed={args.seed})")
    data = generate_dataset(n=args.n, seed=args.seed)
    train, test = train_test_split(data, test_frac=0.2, seed=args.seed)
    print(f"[data]    train={len(train)}  test={len(test)}")

    print(f"[encoder] {args.encoder}")
    encoder = get_encoder(args.encoder)
    model = TrustEvalModel(encoder, hidden=args.hidden)

    print(f"[train]   epochs={args.epochs} lr={args.lr} hidden={args.hidden}")
    stats = model.fit(train, epochs=args.epochs, lr=args.lr, seed=args.seed)
    print(f"[train]   final_batch_loss={stats['final_batch_loss']:.4f}")

    metrics = model.evaluate(test)
    print(f"[eval]    accuracy={metrics['accuracy']:.3f}  "
          f"recall_consistent={metrics['recall_consistent']:.3f}  "
          f"recall_rogue={metrics['recall_rogue']:.3f}  (n={metrics['n']})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save(args.out)
    print(f"[save]    {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
