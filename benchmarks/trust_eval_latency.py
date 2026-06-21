#!/usr/bin/env python3
"""
Per-Hop Trust Evaluation Latency (Milestone 2)
==============================================
Measures the latency the learned ``trust_eval`` adds to a hop, versus the M1
stub. Reports mean / p50 / p95 / p99 over many iterations.

By default it uses the deterministic hashing encoder (no network, reproducible).
Pass ``--encoder sentence-transformers:all-MiniLM-L6-v2`` to measure the real
encoder's cost, or ``--artifact <path>`` to load a trained head.

Usage:
    python benchmarks/trust_eval_latency.py
    python benchmarks/trust_eval_latency.py --encoder sentence-transformers:all-MiniLM-L6-v2 --iters 300
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.a2a.substrate import ChainAgent, originate, process_hop
from agents.a2a.trust import Attestation, DeclaredPurpose, Mu, StubSemanticModel


def _pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _report(name: str, samples_ms: List[float]) -> None:
    print(f"{name:26s} mean={statistics.mean(samples_ms):7.3f}ms  "
          f"p50={_pct(samples_ms,50):7.3f}  p95={_pct(samples_ms,95):7.3f}  "
          f"p99={_pct(samples_ms,99):7.3f}  (n={len(samples_ms)})")


def _time_hops(mu: Mu, agent: ChainAgent, iters: int, warmup: int) -> List[float]:
    samples: List[float] = []
    for i in range(iters + warmup):
        msg = originate("Book a flight from JFK to LAX on 2025-07-01", mu=mu)
        t0 = time.perf_counter()
        process_hop(msg, agent)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            samples.append(dt)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-hop trust_eval latency")
    ap.add_argument("--encoder", default="hashing:256")
    ap.add_argument("--artifact", default=None, help="path to a trained head (else trains one in-process)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    agent = ChainAgent(
        agent_id="airline-agent",
        declared_purpose=DeclaredPurpose(label="flight-search", description="Search and book flights"),
        attestation=Attestation(agent_id="airline-agent"),
        handler=lambda view: "Booked AA123 JFK->LAX conf XJ7F2",
    )

    # Build / load the real model artifact.
    if args.artifact:
        artifact = args.artifact
    else:
        from agents.a2a.semantic.data import generate_dataset, train_test_split
        from agents.a2a.semantic.encoders import get_encoder
        from agents.a2a.semantic.model import TrustEvalModel
        import tempfile
        print(f"[setup] training in-process head (encoder={args.encoder})")
        train, _ = train_test_split(generate_dataset(n=900, seed=0), seed=0)
        m = TrustEvalModel(get_encoder(args.encoder))
        m.fit(train, epochs=30, seed=0)
        artifact = os.path.join(tempfile.mkdtemp(), "head.pt")
        m.save(artifact)

    stub_mu = Mu()  # model_id="stub-semantic-model"
    real_mu = Mu(model_id="trust-eval-v0", params={"artifact": artifact})

    print(f"[run]  iters={args.iters} warmup={args.warmup}\n")
    stub = _time_hops(stub_mu, agent, args.iters, args.warmup)
    real = _time_hops(real_mu, agent, args.iters, args.warmup)

    _report("stub hop (M1)", stub)
    _report("learned hop (M2)", real)
    overhead = statistics.mean(real) - statistics.mean(stub)
    print(f"\nper-hop trust_eval overhead (mean): {overhead:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
