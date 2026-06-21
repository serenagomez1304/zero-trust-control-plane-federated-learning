"""
Per-Message Trust Model — Message Format & Semantic Model
=========================================================
This module defines the data structures for the per-message trust model that
is this paper's security contribution.  Trust is established *per message*
rather than *per principal*: a message carries everything needed to evaluate,
shape, and account for each interaction along its journey through the agent
chain.

A message is the 4-tuple

    m = <content, mu, kappa, sigma>

where (cf. paper_skeleton.tex Sec. 4.1):
  - ``content`` is the payload,
  - ``mu``      (μ) is the bundled semantic model that governs trust decisions,
  - ``kappa``   (κ) is the accumulated context of the message's journey
                (initialized empty at the originating point, carrying the
                user's intent),
  - ``sigma``   (σ) is the cryptographic signature chain authenticating the
                bundle <content, mu, kappa> at each hop.

The semantic model ``mu`` exposes four capabilities, executed by the trusted
compute substrate (the sidecar) on the message's behalf — never by the agents
themselves:
  1. ``trust_eval(purpose, attestation, kappa) -> float`` in [0, 1]
  2. ``transform(content, purpose, kappa) -> Content``  (purpose-tailored view)
  3. ``integrate(response, kappa, purpose) -> Kappa``   (fold response into κ)
  4. ``synthesize(kappa) -> str``                        (final user response)

MILESTONE 1 — STUBS.  At this stage the semantic model is stubbed (see
``StubSemanticModel``); the *structure* and the *pipeline* are real, the
*intelligence* is not.  The real, trained semantic model arrives in Milestone 2.

Open design questions (tracked in CLAUDE.md, not blockers for M1):
  - Is ``purpose`` a string label, a structured object, or an embedding?
    (Here: a small structured object, ``DeclaredPurpose``.)
  - Is ``mu`` a single transformer or a bundle of capability-specific heads?
    (Here: ``Mu`` is a serializable descriptor; ``SemanticModel`` is the
    executable interface with one method per capability — leaving both
    interpretations open.)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("a2a.trust")


# =============================================================================
# content — the payload
# =============================================================================

class Content(BaseModel):
    """The message payload.

    ``transform`` derives a purpose-tailored *view* of this content for each
    agent; the original ``content`` carried in the message is never mutated, so
    an agent only ever sees its view, never the raw payload or other agents'
    views.
    """
    payload: str = Field(..., description="Primary payload (e.g. the user request text)")
    data: Dict[str, Any] = Field(default_factory=dict, description="Optional structured payload")


# =============================================================================
# Per-agent inputs to trust evaluation — declared purpose & attestation
# =============================================================================

class DeclaredPurpose(BaseModel):
    """The purpose an agent declares for the interaction (``p_a``).

    ``trust_eval`` checks this declared purpose against the message's intent and
    accumulated context.  Modeled as a small structured object for M1; whether
    it should ultimately be a label, a structured object, or an embedding is an
    open design question.
    """
    label: str = Field(..., description="Short purpose identifier, e.g. 'flight-search'")
    description: str = Field(default="", description="Human-readable purpose statement")


class Attestation(BaseModel):
    """An agent's attestation (``alpha_a``), nominally signed by a trusted attester.

    The fields below are a minimal M1 stub.  What an attestation should actually
    contain (identity, signed code hash, recent failure rate, declared purpose)
    is an open design question tracked in CLAUDE.md.
    """
    agent_id: str = Field(..., description="Identity of the attesting agent")
    code_hash: Optional[str] = Field(default=None, description="Signed hash of the agent's code/model")
    signed_by: Optional[str] = Field(default=None, description="Identity of the trusted attester")
    recent_failure_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Recent observed failure rate, if known"
    )


# =============================================================================
# kappa — accumulated context
# =============================================================================

class ContextEntry(BaseModel):
    """One agent's contribution folded into the accumulated context by ``integrate``.

    Keyed by the declared purpose of the hop (the spec's ``integrate`` receives
    the purpose, not the agent identity), so context is attributed to *what the
    interaction was for* rather than to the raw agent.
    """
    purpose: str = Field(..., description="Declared purpose label of the hop that produced this")
    response: str = Field(..., description="The agent's response, as integrated")


class Kappa(BaseModel):
    """The running state of the message's journey through the agent chain.

    Initialized at the originating point with the user's ``intent`` and no
    entries; each hop appends one entry via ``integrate``.
    """
    intent: str = Field(..., description="The originating user intent")
    entries: List[ContextEntry] = Field(
        default_factory=list, description="Per-hop integrated responses, in chain order"
    )


# =============================================================================
# sigma — signature chain
# =============================================================================

class Sigma(BaseModel):
    """The signature chain authenticating <content, mu, kappa> at each hop.

    Each hop appends a signature over the updated bundle (the substrate
    "resigns with updated context").  M1 uses keyless hash-based stub
    signatures computed in ``substrate.py``; real per-hop signing is later work.
    """
    chain: List[str] = Field(default_factory=list, description="Signatures, one appended per hop")


# =============================================================================
# mu — the bundled semantic model (descriptor + executable interface)
# =============================================================================

class Mu(BaseModel):
    """Serializable descriptor of the bundled semantic model carried in the message.

    This is the *data* that travels with the message; the *behavior* is the
    ``SemanticModel`` the substrate resolves from it (see ``resolve_mu``).  For
    M1 the descriptor identifies the stub; M2 will carry real model parameters
    (or a reference to them).
    """
    # ``model_`` is a Pydantic protected namespace — opt out so ``model_id`` is allowed.
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(default="stub-semantic-model", description="Identifier of the semantic model")
    version: str = Field(default="0.1.0-stub", description="Semantic model version")
    bundle: List[str] = Field(
        default_factory=lambda: ["trust_eval", "transform", "integrate", "synthesize"],
        description="Capabilities provided by this semantic model",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Opaque model parameters / references (empty for the M1 stub)"
    )


class SemanticModel(ABC):
    """Executable interface for the four capabilities of ``mu``.

    Implementations run on the trusted compute substrate, never inside agents.
    """

    @abstractmethod
    def trust_eval(self, purpose: DeclaredPurpose, attestation: Attestation, kappa: Kappa) -> float:
        """Return a trust value in [0, 1] for letting this agent process the message."""

    @abstractmethod
    def transform(self, content: Content, purpose: DeclaredPurpose, kappa: Kappa) -> Content:
        """Return the purpose-tailored view of ``content`` for this agent."""

    @abstractmethod
    def integrate(self, response: str, kappa: Kappa, purpose: DeclaredPurpose) -> Kappa:
        """Fold an agent's ``response`` into a new accumulated context."""

    @abstractmethod
    def synthesize(self, kappa: Kappa) -> str:
        """Synthesize the final user-facing response from accumulated context."""


class StubSemanticModel(SemanticModel):
    """Milestone 1 stub semantic model — structure-faithful, intelligence-free.

    Behaviors are exactly as specified for M1:
      - ``trust_eval``  always returns 1.0 (everything is trusted)
      - ``transform``   returns content unchanged (no purpose tailoring yet)
      - ``integrate``   appends the response to accumulated context
      - ``synthesize``  returns the latest response

    The real model arrives in Milestone 2.
    """

    def trust_eval(self, purpose: DeclaredPurpose, attestation: Attestation, kappa: Kappa) -> float:
        return 1.0

    def transform(self, content: Content, purpose: DeclaredPurpose, kappa: Kappa) -> Content:
        return content

    def integrate(self, response: str, kappa: Kappa, purpose: DeclaredPurpose) -> Kappa:
        updated = kappa.model_copy(deep=True)
        updated.entries.append(ContextEntry(purpose=purpose.label, response=response))
        return updated

    def synthesize(self, kappa: Kappa) -> str:
        if not kappa.entries:
            return ""
        return kappa.entries[-1].response


# Default location of the trained trust_eval artifact (written by
# agents.a2a.semantic.train).  A message's ``mu.params["artifact"]`` overrides it.
DEFAULT_ARTIFACT = os.path.join(os.path.dirname(__file__), "semantic", "artifacts", "trust_eval_v0.pt")

# Cache resolved real models by artifact path so the encoder/head load once.
_REAL_MODEL_CACHE: Dict[str, SemanticModel] = {}


def resolve_mu(mu: Mu) -> SemanticModel:
    """Resolve a message's ``mu`` descriptor to its executable semantic model.

    - A descriptor whose ``model_id`` starts with ``"stub"`` resolves to the M1
      stub (``StubSemanticModel``).
    - Otherwise the M2 real model (``RealSemanticModel`` wrapping a trained
      ``TrustEvalModel``) is loaded from ``mu.params["artifact"]`` or
      ``DEFAULT_ARTIFACT``.  If the artifact or ML deps are unavailable, this
      logs a warning and falls back to the stub so the pipeline still runs.
    """
    if mu.model_id.startswith("stub"):
        return StubSemanticModel()

    artifact = mu.params.get("artifact") or DEFAULT_ARTIFACT
    cached = _REAL_MODEL_CACHE.get(artifact)
    if cached is not None:
        return cached

    try:
        # Lazy import: keeps torch / sentence-transformers out of the base import path.
        from agents.a2a.semantic.model import RealSemanticModel, TrustEvalModel

        real = RealSemanticModel(TrustEvalModel.load(artifact))
        _REAL_MODEL_CACHE[artifact] = real
        return real
    except Exception as exc:  # missing artifact, missing torch, etc.
        logger.warning("resolve_mu: falling back to stub for mu=%s (%s)", mu.model_id, exc)
        return StubSemanticModel()


# =============================================================================
# m = <content, mu, kappa, sigma>
# =============================================================================

class TrustMessage(BaseModel):
    """A message in the per-message trust model: ``m = <content, mu, kappa, sigma>``."""
    content: Content
    mu: Mu
    kappa: Kappa
    sigma: Sigma
