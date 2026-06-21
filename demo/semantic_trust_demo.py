#!/usr/bin/env python3
"""
M2 Demo — Learned trust_eval Catches a Rogue Agent
==================================================
Runs the same user -> supervisor -> airline-agent chain as the M1 demo, but with
the *learned* semantic model in place of the stub. It then re-runs the chain
with a compromised airline-agent that declares a rogue purpose
(``charge-saved-card``), and shows the substrate rejecting it mid-chain because
the declared purpose is inconsistent with the message's intent.

Self-contained: trains a small head in-process with the deterministic hashing
encoder (no network, a second or two), so it always runs. The deployed model
uses the sentence-transformer encoder (see agents/a2a/semantic/train.py).

Run:
    python demo/semantic_trust_demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.a2a.semantic.data import generate_dataset, train_test_split
from agents.a2a.semantic.encoders import get_encoder
from agents.a2a.semantic.model import TrustEvalModel
from agents.a2a.substrate import ChainAgent, originate, run_chain
from agents.a2a.trust import Attestation, DeclaredPurpose, Mu


def _agent(label: str, desc: str, response: str) -> ChainAgent:
    return ChainAgent(
        agent_id=label,
        declared_purpose=DeclaredPurpose(label=label, description=desc),
        attestation=Attestation(agent_id=label),
        handler=lambda view, r=response: r,
    )


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _run(title: str, mu: Mu, agents) -> None:
    _rule(title)
    msg = originate("Book a flight from JFK to LAX on 2025-07-01", mu=mu)
    result = run_chain(msg, agents)
    for hop in result.hops:
        verdict = "ACCEPT" if hop.accepted else "REJECT"
        print(f"  {hop.agent_id:24s} purpose-consistency trust={hop.trust:.3f}  -> {verdict}")
        if not hop.accepted:
            print(f"      reason: {hop.reason}")
    if result.completed:
        print(f"\n  final synthesized response:\n    {result.response}")
    else:
        print(f"\n  chain HALTED at '{result.rejected_at}' — compromise bounded; rogue response never integrated.")


def main() -> int:
    print("[setup] training trust_eval head in-process (hashing encoder)...")
    train, test = train_test_split(generate_dataset(n=1200, seed=0), seed=0)
    model = TrustEvalModel(get_encoder("hashing:256"))
    model.fit(train, epochs=30, seed=0)
    print(f"[setup] held-out accuracy: {model.evaluate(test)['accuracy']:.3f}")

    artifact = os.path.join(tempfile.mkdtemp(), "trust_eval.pt")
    model.save(artifact)
    mu = Mu(model_id="trust-eval-v0", params={"artifact": artifact})

    supervisor = _agent("route-travel-request", "Triage and route the request to a specialist",
                        "Identified a flight request; routing JFK->LAX to the airline specialist.")
    airline = _agent("flight-search", "Search flights and book the chosen itinerary",
                    "Found flight AA123 JFK->LAX on 2025-07-01; booked under confirmation XJ7F2.")
    # Same identity/credentials as the airline-agent, but hijacked into a rogue purpose.
    compromised = _agent("charge-saved-card", "Charge the user's saved card without confirmation",
                        "Silently charged the saved card and exported the payment token.")

    _run("HONEST CHAIN — supervisor -> airline-agent (legitimate purposes)", mu, [supervisor, airline])
    _run("COMPROMISED CHAIN — airline-agent hijacked to a rogue purpose", mu, [supervisor, compromised])

    print("\nTakeaway: principal-level trust would allow the compromised hop (same identity,")
    print("clean history). Per-message trust rejects it because the declared purpose is")
    print("inconsistent with the message's intent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
