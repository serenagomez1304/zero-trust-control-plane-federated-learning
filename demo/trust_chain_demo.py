#!/usr/bin/env python3
"""
End-to-End Demo — Per-Message Trust Model (Milestone 1)
=======================================================
Runs a single user request through one agent chain

    user  ->  supervisor  ->  airline-agent

using the per-message trust model: each message carries
``m = <content, mu, kappa, sigma>``, and at every hop the trusted substrate
runs the four-stage pipeline (verify -> trust_eval -> transform -> integrate)
on the message's behalf, then synthesizes the final response from accumulated
context.

This is the Milestone 1 demo: the semantic model ``mu`` is stubbed, so trust
always passes, the view equals the content, and synthesis returns the latest
response.  The point is to show the *structure and flow*, not the intelligence
(which arrives in M2).  Everything runs in-process; no Docker, MCP, or LLM.

Run:
    python demo/trust_chain_demo.py
"""

from __future__ import annotations

import os
import sys

# Allow running directly from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.a2a.substrate import ChainAgent, originate, run_chain
from agents.a2a.trust import Attestation, Content, DeclaredPurpose


# -----------------------------------------------------------------------------
# Stub agents (untrusted).  Each returns a canned response for its view.
# -----------------------------------------------------------------------------

def supervisor_handler(view: Content) -> str:
    return "Identified a flight request; routing JFK->LAX on 2025-07-01 to the airline specialist."


def airline_handler(view: Content) -> str:
    return "Found flight AA123 JFK->LAX on 2025-07-01 at $320; booked under confirmation XJ7F2."


SUPERVISOR = ChainAgent(
    agent_id="supervisor",
    declared_purpose=DeclaredPurpose(
        label="route-travel-request",
        description="Triage the user request and route it to the right specialist agent.",
    ),
    attestation=Attestation(agent_id="supervisor", signed_by="attester", recent_failure_rate=0.0),
    handler=supervisor_handler,
)

AIRLINE_AGENT = ChainAgent(
    agent_id="airline-agent",
    declared_purpose=DeclaredPurpose(
        label="flight-search-and-booking",
        description="Search flights and book the chosen itinerary.",
    ),
    attestation=Attestation(agent_id="airline-agent", signed_by="attester", recent_failure_rate=0.0),
    handler=airline_handler,
)


def _rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    intent = "Book me a flight from JFK to LAX on 2025-07-01."

    _rule("ORIGINATE  —  user creates m = <content, mu, kappa, sigma>")
    message = originate(intent)
    print(f"intent (kappa): {message.kappa.intent}")
    print(f"content        : {message.content.payload}")
    print(f"mu             : {message.mu.model_id} v{message.mu.version} bundle={message.mu.bundle}")
    print(f"sigma chain    : {len(message.sigma.chain)} signature(s)")

    _rule("RUN CHAIN  —  user -> supervisor -> airline-agent")
    result = run_chain(message, [SUPERVISOR, AIRLINE_AGENT])

    for i, hop in enumerate(result.hops, start=1):
        print(f"\n[hop {i}] {hop.agent_id}")
        print(f"  trust_eval : {hop.trust:.2f}  (threshold {hop.threshold:.2f})  -> "
              f"{'ACCEPT' if hop.accepted else 'REJECT'}")
        if hop.accepted:
            print(f"  view shown : {hop.view.payload!r}")
            print(f"  response   : {hop.response!r}")
            print(f"  kappa now  : {[e.purpose for e in hop.message.kappa.entries]}")
            print(f"  sigma chain: {len(hop.message.sigma.chain)} signatures (resigned)")
        else:
            print(f"  reason     : {hop.reason}")

    _rule("SYNTHESIZE  —  final response built from accumulated context")
    print(f"completed       : {result.completed}")
    print(f"accumulated ctx : ")
    for e in result.final_message.kappa.entries:
        print(f"    - [{e.purpose}] {e.response}")
    print(f"\nfinal response to user:\n    {result.response}")

    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
