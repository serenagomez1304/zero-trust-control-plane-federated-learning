# Milestone 2 — Real Semantic Model (design note)

M2 replaces the M1 stubs in `mu` with a real implementation. This note records
the design and points at the code. It corresponds to the brief §5 and §4.2/§6
of `docs/paper_skeleton.tex`.

## Scope

Of the four `mu` capabilities, **only `trust_eval` is a learned model** — it is
the security decision the paper rests on and the model that federated learning
trains in M4. The other three are upgraded from stubs to **real-but-deterministic**
implementations (no generative model needed):

| Capability   | M1 stub                 | M2 |
|--------------|-------------------------|----|
| `trust_eval` | always 1.0              | **learned** consistency classifier → [0,1] |
| `transform`  | content unchanged       | deterministic purpose-scoped projection of content |
| `integrate`  | append response         | append + minimal long-context trimming |
| `synthesize` | latest response         | deterministic extractive composition over κ |

## The learned `trust_eval`

**Question.** Is the next agent's *declared purpose* consistent with the
message's *intent* and *accumulated context*? Output is a trust value in [0,1];
the substrate rejects a hop when it falls below `TRUST_THRESHOLD` (0.5).

**Architecture (frozen encoder + trainable head).**
- A pretrained sentence-transformer encoder (`all-MiniLM-L6-v2`, 384-d) embeds
  three texts: the intent (`κ.intent`), the purpose (`label: description`), and
  the accumulated context (prior responses in κ). The encoder is **frozen**.
- Features: `concat(e_intent, e_purpose, e_context, e_intent ⊙ e_purpose)`.
- A small MLP head (`Linear → ReLU → Dropout → Linear → sigmoid`) outputs the
  trust score. **Only the head is trained**, so M4 federates a small weight
  vector across deployments.

The encoder is pluggable (`agents/a2a/semantic/encoders.py`): the real
sentence-transformer is the default for training/demo; a deterministic,
no-network **hashing encoder** is used for fast/offline tests and reproducible
FL simulation.

**Training data (synthetic).** `agents/a2a/semantic/data.py` generates
`(intent, declared_purpose, context) → consistent?` tuples from the travel
testbed's agent skills and intents:
- **Consistent (label 1):** purpose matches the intent's domain
  (flight intent → `flight-search`/`flight-booking`).
- **Inconsistent (label 0):** the rogue-agent signal —
  - cross-domain mismatch (flight intent → `hotel-booking`),
  - escalation/exfiltration purposes (`charge-saved-card-without-confirmation`,
    `exfiltrate-passenger-data`, `override-approval-policy`, …).

This is exactly what a hijacked/substituted/supply-chain-compromised agent would
declare, and what principal-level trust misses. The generator is reused by M3
(attack harness) and M4 (per-deployment FL partitions).

## Code map
- `agents/a2a/semantic/encoders.py` — encoder interface + sentence-transformer / hashing impls.
- `agents/a2a/semantic/data.py` — synthetic dataset generator.
- `agents/a2a/semantic/model.py` — `TrustEvalModel` (torch head) + `RealSemanticModel` (implements `SemanticModel`).
- `agents/a2a/semantic/train.py` — CLI: generate data → train head → save artifact → report metrics.
- `agents/a2a/trust.py::resolve_mu` — returns the real model when an artifact is configured, else the M1 stub.
- `benchmarks/trust_eval_latency.py` — per-hop latency measurement.

## Deferred (see TODO.md)
Over-the-wire substrate (M3/M6), real `sigma` signing, full-encoder fine-tuning,
generative `transform`/`synthesize`, attestation actually consumed by `trust_eval`.
