# TODO / Open Questions / Deferred Items

Tracking file requested by the handoff brief (§6). Open questions, deferred
work, and known issues. Update as work progresses.

## Milestone status
- [x] **M1 — end-to-end message flow with stubs.** Message format
  (`content, mu, kappa, sigma`), 4-stage substrate pipeline, in-process demo,
  tests. See CLAUDE.md.
- [ ] **M2 — real semantic model (in progress).** Learned `trust_eval`
  (frozen encoder + trainable head); deterministic `transform`/`integrate`/
  `synthesize`; synthetic data; per-hop latency.
- [ ] M3 — rogue-agent attack harness.
- [ ] M4 — federated learning setup (Flower; FedAvg/FedProx/SCAFFOLD/FedNova).
- [ ] M5 — heterogeneity characterization.
- [ ] M6 — overhead and ablation.

## M2 design decisions (locked with advisor)
- **Substrate:** frozen pretrained sentence-transformer encoder + small
  trainable MLP head. Only the head is trained (and federated in M4).
- **Learning scope:** only `trust_eval` is a learned model. `transform`,
  `integrate`, `synthesize` are real-but-deterministic (no generative model).
- **Training data:** synthetic `(intent, declared_purpose, context) → consistent?`
  tuples generated from the travel testbed's skills/intents.

## Deferred (documented, not dropped)
- **Over-the-wire substrate.** Pipeline runs in-process; the real C# `zta-sidecar`
  (or a Python sidecar) fronting agents with the trust envelope in A2A metadata
  is needed for the rogue-agent isolation demo (M3) and overhead numbers (M6).
- **Real `sigma` signing.** Keyless SHA-256 stub today; per-hop crypto signing
  (keys/attester) later.
- **Full-encoder fine-tuning.** M2 freezes the encoder; fine-tuning the whole
  DistilBERT-class model end-to-end is a possible later upgrade.
- **Generative `transform`/`synthesize`.** Deterministic for now; could become
  learned/LLM-backed if the paper needs it.

## Open questions (from brief §7; revisit as they bite)
- "Purpose" representation: currently a small structured object
  (`DeclaredPurpose{label, description}`) whose text is encoded. Embedding vs
  richer structure still open.
- `mu` as one shared model vs a bundle of capability heads: M2 only learns
  `trust_eval`, so unresolved for the other capabilities.
- Context summarization for long chains: `integrate` does minimal trimming;
  needs a real strategy before long-chain experiments.
- Attestation contents: `Attestation` is a minimal stub
  (`agent_id, code_hash?, signed_by?, recent_failure_rate?`); `trust_eval`
  does not yet consume it.

## Known cleanups (low priority)
- `agents/a2a/models.py` still defines an unused `TRUST_SCORE_TOO_LOW` error
  code; `tests/test_a2a.py` uses `x-trust-score` in a generic header-propagation
  test. Harmless, outside the active path — scrub opportunistically.
