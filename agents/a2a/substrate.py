"""
Trusted Compute Substrate — Per-Hop Trust Pipeline
==================================================
The substrate is the trusted compute environment (the sidecar) on which a
message's bundled semantic model ``mu`` executes.  Agents are untrusted
services; the substrate is trusted.  This module implements the per-hop
pipeline from paper_skeleton.tex Sec. 4.2 (Algorithm 1) plus final synthesis
(Sec. 4.3).

Per-hop pipeline, executed by the substrate on the message's behalf:

    verify  ->  trust_eval  ->  transform  ->  (agent runs)  ->  integrate

  1. verify       — verify sigma over <content, mu, kappa>
  2. trust_eval   — t = mu.trust_eval(p_a, alpha_a, kappa); reject if t < tau
  3. transform    — view = mu.transform(content, p_a, kappa)   (agent sees only this)
  4. agent runs   — r = agent(view)                            (untrusted)
  5. integrate    — kappa' = mu.integrate(r, kappa, p_a); resign -> sigma'

At the end of the chain the substrate computes ``synthesize(kappa)``, producing
the user-facing response from accumulated context rather than from any single
agent.

Key invariants (the source of the security property):
  - The agent is invoked only on its ``view``; ``message.content`` is never
    mutated and never handed to an agent directly.
  - The agent's response enters the message only through ``integrate``.
  - The agent never produces the user-facing output; ``synthesize`` does.

MILESTONE 1 — IN-PROCESS, STUBBED.
  - ``mu`` is the stub (see trust.StubSemanticModel).
  - Signatures are keyless SHA-256 hashes over the canonical bundle — enough to
    make ``verify`` meaningful and tamper-detecting, but not real signatures.
  - The pipeline runs in-process; agents are plain callables.

DEFERRED (documented future work, see CLAUDE.md milestones):
  - Running this pipeline in the *real* substrate over the wire — the C#
    ``zta-sidecar`` (or a Python sidecar process) fronting each agent, with the
    trust envelope propagated in A2A message metadata — is needed for the
    rogue-agent attack harness (M3, to demonstrate the isolation boundary) and
    overhead measurement (M6).  M1 deliberately stays in-process.
  - Real per-hop cryptographic signing of ``sigma`` (keys / attester) replaces
    the stub hash signatures.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from agents.a2a.trust import (
    Attestation,
    Content,
    DeclaredPurpose,
    Kappa,
    Mu,
    Sigma,
    TrustMessage,
    resolve_mu,
)

logger = logging.getLogger("a2a.substrate")

# Trust threshold tau.  A hop is rejected when trust_eval returns < TRUST_THRESHOLD.
# The M1 stub always returns 1.0, so the accept path is exercised by default; the
# reject path is reachable via a higher threshold or a non-stub semantic model.
TRUST_THRESHOLD = 0.5


# =============================================================================
# Agents in the chain (untrusted)
# =============================================================================

# An agent handler receives its purpose-tailored view and returns a response.
AgentHandler = Callable[[Content], str]


@dataclass
class ChainAgent:
    """An untrusted agent the substrate mediates a hop to.

    ``declared_purpose`` and ``attestation`` are inputs to ``trust_eval``.  The
    agent is invoked (``__call__``) only on the view produced by ``transform``.
    """
    agent_id: str
    declared_purpose: DeclaredPurpose
    attestation: Attestation
    handler: AgentHandler

    def __call__(self, view: Content) -> str:
        return self.handler(view)


# =============================================================================
# Signature stubs (sigma)
# =============================================================================
#
# M1 signatures are keyless SHA-256 digests over the canonical JSON of the
# signed bundle <content, mu, kappa>.  This is not authentication — anyone can
# recompute it — but it makes ``verify`` tamper-detecting and the resign step
# meaningful, which is what M1 needs.  Real signing is later work.

def _canonical(*models) -> str:
    """Canonical (sorted-key) JSON join of pydantic models, for stable hashing."""
    return "|".join(
        json.dumps(m.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for m in models
    )


def _sign(content: Content, mu: Mu, kappa: Kappa) -> str:
    digest = hashlib.sha256(_canonical(content, mu, kappa).encode("utf-8")).hexdigest()
    return f"stub-sig:{digest}"


def verify_signature(message: TrustMessage) -> bool:
    """Verify the latest signature matches the current <content, mu, kappa> bundle."""
    if not message.sigma.chain:
        return False
    expected = _sign(message.content, message.mu, message.kappa)
    return message.sigma.chain[-1] == expected


# =============================================================================
# Origination
# =============================================================================

def originate(
    intent: str,
    payload: Optional[str] = None,
    *,
    mu: Optional[Mu] = None,
    data: Optional[dict] = None,
) -> TrustMessage:
    """Create the initial message at the originating point.

    ``kappa`` starts with the user's ``intent`` and no entries; ``sigma`` carries
    the first signature over the initial bundle.
    """
    content = Content(payload=payload if payload is not None else intent, data=data or {})
    mu = mu or Mu()
    kappa = Kappa(intent=intent)
    return TrustMessage(content=content, mu=mu, kappa=kappa, sigma=Sigma(chain=[_sign(content, mu, kappa)]))


# =============================================================================
# Per-hop pipeline
# =============================================================================

@dataclass
class HopResult:
    """Outcome of a single hop."""
    agent_id: str
    accepted: bool
    trust: float
    threshold: float
    reason: str = ""
    view: Optional[Content] = None
    response: Optional[str] = None
    message: Optional[TrustMessage] = None  # the resigned message, if accepted


def process_hop(
    message: TrustMessage,
    agent: ChainAgent,
    threshold: float = TRUST_THRESHOLD,
) -> HopResult:
    """Run the four-stage pipeline for one hop (Algorithm 1).

    Returns a rejected ``HopResult`` (with no updated message) on signature
    failure or insufficient trust; otherwise an accepted ``HopResult`` carrying
    the resigned message with updated accumulated context.
    """
    # 1. Verify signature over <content, mu, kappa>.
    if not verify_signature(message):
        logger.warning("hop|%s|reject|signature verification failed", agent.agent_id)
        return HopResult(agent.agent_id, False, 0.0, threshold, reason="signature verification failed")

    mu = resolve_mu(message.mu)

    # 2. Trust evaluation.
    trust = mu.trust_eval(agent.declared_purpose, agent.attestation, message.kappa)
    if trust < threshold:
        logger.info("hop|%s|reject|trust=%.3f < tau=%.3f", agent.agent_id, trust, threshold)
        return HopResult(
            agent.agent_id, False, trust, threshold,
            reason=f"trust {trust:.3f} below threshold {threshold:.3f}",
        )

    # 3. Purpose-driven transform — the only thing the agent ever sees.
    view = mu.transform(message.content, agent.declared_purpose, message.kappa)

    # 4. The untrusted agent processes its view.
    response = agent(view)

    # 5. Integrate the response into accumulated context and resign.
    kappa2 = mu.integrate(response, message.kappa, agent.declared_purpose)
    sigma2 = Sigma(chain=[*message.sigma.chain, _sign(message.content, message.mu, kappa2)])
    new_message = TrustMessage(content=message.content, mu=message.mu, kappa=kappa2, sigma=sigma2)

    logger.info("hop|%s|accept|trust=%.3f", agent.agent_id, trust)
    return HopResult(
        agent.agent_id, True, trust, threshold,
        view=view, response=response, message=new_message,
    )


def synthesize(message: TrustMessage) -> str:
    """Synthesize the final user-facing response from accumulated context (Sec. 4.3)."""
    return resolve_mu(message.mu).synthesize(message.kappa)


# =============================================================================
# Chain driver
# =============================================================================

@dataclass
class ChainResult:
    """Outcome of running a message through a chain of agents."""
    completed: bool
    final_message: TrustMessage
    response: Optional[str]
    hops: List[HopResult] = field(default_factory=list)
    rejected_at: Optional[str] = None


def run_chain(
    message: TrustMessage,
    agents: List[ChainAgent],
    threshold: float = TRUST_THRESHOLD,
) -> ChainResult:
    """Drive a message through a chain of agents, hop by hop, then synthesize.

    Stops at the first rejected hop (returning ``completed=False``); otherwise
    runs every hop and synthesizes the final response from accumulated context.
    """
    current = message
    hops: List[HopResult] = []
    for agent in agents:
        result = process_hop(current, agent, threshold)
        hops.append(result)
        if not result.accepted:
            return ChainResult(False, current, None, hops, rejected_at=agent.agent_id)
        current = result.message  # type: ignore[assignment]  # accepted => message set

    return ChainResult(True, current, synthesize(current), hops)
