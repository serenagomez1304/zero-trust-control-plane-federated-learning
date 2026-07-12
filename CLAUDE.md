# CLAUDE.md

This file is read automatically by Claude Code at every session start. It captures the persistent context for this project. Update it as the project evolves; don't let it grow stale.

---

## What this project is

This repo is the codebase for a research paper provisionally titled **"Federated Learning of Purpose- and Context-Driven Trust Models for Multi-Agent AI Communication."** It is being submitted to a *Machine Learning* journal special issue on federated learning for critical applications.

The repo started as a copy of an earlier project (an existing multi-agent zero-trust testbed built for a NeurIPS submission). The infrastructure — agents, MCP servers, sidecars, mTLS, the Docker testbed — is being reused. The **trust model itself is fundamentally different** and is what this paper contributes.

### The shift in trust model

**Old model (the original NeurIPS work, not what this paper is about):** Trust is computed by a central trust-scorer service. Sidecars query the scorer; the scorer fuses identity, behavioral, and LLM-layer evidence using Cumulative Belief Fusion (Subjective Logic). The trust decision is made by infrastructure external to the message.

**New model (THIS paper):** Trust is established per-message rather than per-principal. Each message carries:
- A **bundled semantic model** (`mu`) — a small ML model that handles trust evaluation, transformation, response integration, and synthesis
- An **accumulated context** (`kappa`) — the running state of the message's journey through the agent chain

At every hop, `mu` executes on the **trusted compute substrate (the sidecar)** — NOT in the agent itself — and:
1. Evaluates whether the next agent's declared purpose is consistent with the message's intent and accumulated context
2. If trust passes, transforms the message into a purpose-tailored view for that agent
3. Integrates the agent's response into accumulated context
4. At the end of the chain, synthesizes the final response to the user from the accumulated context, not from any single agent

There is **no central trust scorer**, no central PDP, no external policy engine. The platform (sidecars + supporting infra) is trusted compute; the agents are untrusted services.

### The two contributions

1. **Purpose- and context-driven per-message trust model** (the security contribution)
2. **Federated learning of the semantic models** that drive these trust decisions, since training data is distributed and sensitive (the FL contribution)

---

## What we're trying to defend against

The threat model centers on **rogue or compromised agents**:
- An agent that was previously well-behaved but has been hijacked by prompt injection
- An agent substituted by an attacker carrying the same credentials
- An agent compromised via supply chain (model or code replaced)

Today's principal-based trust catches none of these because the agent's identity and behavioral history still look fine. The per-message model catches them because the message evaluates each agent's declared purpose against the message's intent and accumulated context, and the agent only ever sees the purpose-tailored view (so compromise is bounded).

---

## Current milestone

**Milestone 5 (next): heterogeneity characterization.** M1–M4 are complete (see below). M5 measures the non-IID distances between deployment marginals (agent-population, task-distribution, threat-profile) — e.g. Wasserstein/KL between the client partitions from M4 — and correlates heterogeneity with FL performance. Produces the data for §5.3.

---

### ✅ Milestone 4: federated learning of the trust_eval head (complete)

Trains the head across K simulated deployments holding non-IID slices of the synthetic trust data; compares aggregation strategies to a centralized reference (Table 2). **Decision (locked with advisor):** a **hand-rolled FL simulation** (not Flower) — full control over all four strategies, deterministic, no heavy dependency. Only the head is federated (encoder frozen).

- Code: `agents/a2a/federated/` (`partition.py` = Dirichlet(α) non-IID split; `strategies.py` = FedAvg/FedProx/SCAFFOLD/FedNova on the head's weight vector; `simulate.py` = FL loop + centralized baseline). Design note: `docs/m4_design.md`.
- CLI: `evaluation/run_federated.py` → `evaluation/results/federated_results.{csv,json}`.
- Results (synthetic, hashing encoder, K=5): centralized ≈1.00; FL matches on IID/mild non-IID; under strong skew (α=0.1) strategies diverge — FedNova ≈1.00, FedAvg ≈0.99, FedProx/SCAFFOLD ≈0.86–0.87. Tests: `tests/test_federated.py`. Full suite **78 passing**.
- **Deferred:** larger K sweeps + multi-seed variance; real-encoder FL run; formal heterogeneity metrics (that's M5); optional DP-SGD.

---

### ✅ Milestone 3: rogue-agent attack harness (complete)

Implements the three attack classes from paper §3.3 and produces the §6.1 Table 1 data. Each scenario runs through a **principal-based baseline** (trusts any known agent with clean history; no purpose-scoped view) and **our per-message model**.

- **Decision (locked with advisor):** supply-chain compromise (where the declared purpose still looks legitimate) is caught by an **attestation code-hash gate**; substitution and prompt injection are caught by the learned `trust_eval` (rogue declared purpose). Both are part of "ours".
- Substrate hooks (backward-compatible): `process_hop`/`run_chain` take an optional `attestation_registry` (the gate) and an optional `model` override (baseline vs. ours). See `agents/a2a/substrate.py`.
- Harness: `evaluation/attacks.py` (baseline foil, scenario generators, `evaluate`/`format_table`); CLI `evaluation/run_attacks.py` writes `evaluation/results/attack_results.{csv,json}`. Injection scenarios reuse `test-driver/prompts/prompt_injection.json`.
- Results (synthetic/controlled): baseline ASR **1.00** and exposure **1.00** on every class; ours **0.00 / 0.00**. Tests: `tests/test_attacks.py` (Table 1 + mechanism-attribution + bounded-exposure). Full suite **68 passing**.
- **Deferred:** rogue detection via context-integration drift (secondary signal); over-the-wire isolation demo (needs the real substrate); `trust_eval` generalizing to unseen rogue purposes. See `TODO.md`.

---

### ✅ Milestone 2: real semantic model (complete)

Replaced the M1 stubs with a real implementation. **Decisions (locked with advisor):** only `trust_eval` is a learned model (frozen sentence-transformer encoder + trainable MLP head — M4 federates just the head); `transform`/`integrate`/`synthesize` are real-but-deterministic. Trained on synthetic `(intent, declared_purpose, context) → consistent?` tuples generated from the testbed scenarios.

- Code: `agents/a2a/semantic/` (`encoders.py`, `data.py`, `model.py`, `train.py`), wired via `agents/a2a/trust.py::resolve_mu`. Design note: `docs/m2_design.md`.
- Trained artifact committed: `agents/a2a/semantic/artifacts/trust_eval_v0.pt` (head only; encoder loaded by spec). Retrain: `python -m agents.a2a.semantic.train`.
- Results: 100% held-out accuracy on synthetic data; legit purpose ≈0.97 trust, rogue purpose ≈0.00 → rejected mid-chain. Demo: `demo/semantic_trust_demo.py`. Latency (`benchmarks/trust_eval_latency.py`): ~16 ms/hop real encoder, ~0.06 ms/hop hashing encoder (CPU).
- Tests: `tests/test_semantic_model.py` (offline hashing encoder, skips without the `ml` extra). Full suite 61 passing.
- **Deferred:** full-encoder fine-tuning; generative `transform`/`synthesize`; `trust_eval` consuming the attestation. See `TODO.md`.

---

### ✅ Milestone 1: end-to-end message flow with stubs (complete)

Goal: get one message moving through the system with the new structure (`content`, `mu`, `kappa`, `sigma`), the new 4-stage sidecar pipeline (verify → trust_eval → transform → integrate), and an end-to-end demo. The semantic model is stubbed at this stage:
- `mu.trust_eval` always returns 1.0
- `mu.transform` returns content unchanged
- `mu.integrate` appends response to context
- `mu.synthesize` returns the latest response

Acceptance criteria:
- [x] Single user request flows end-to-end through one agent chain (user → supervisor → airline-agent) — `demo/trust_chain_demo.py`, in-process
- [x] Message format is documented and the schema is in code — `agents/a2a/trust.py` + `docs/message_format.md`
- [x] Sidecar middleware pipeline executes the 4 stages — `agents/a2a/substrate.py` (`process_hop`: verify → trust_eval → transform → integrate, plus `synthesize`)
- [x] All previous trust-scorer references are removed from the active code path
- [x] Tests in `tests/test_a2a.py` pass with the new message format — 42 passing (23 original + 19 new)

**M1 implementation decisions (for continuity):**
- The 4-stage pipeline is implemented in a **Python substrate module** (`agents/a2a/substrate.py`), not the C# `zta-sidecar`. This is what `tests/test_a2a.py` exercises and what the demo runs — no Docker/MCP/LLM needed for the stub milestone.
- The end-to-end path is an **in-process stub chain** (`ChainAgent` callables with canned purposes/responses), not real A2A servers over HTTP.
- The semantic model `mu` is the **stub** (`StubSemanticModel`): `trust_eval`→1.0, `transform`→unchanged, `integrate`→append, `synthesize`→latest.
- `sigma` signatures are **keyless SHA-256 stubs** — tamper-detecting but not authentication.

**Deferred from M1 (documented future work, not dropped):**
- Running the pipeline in the **real over-the-wire substrate** (C# `zta-sidecar` or a Python sidecar process, trust envelope in A2A metadata) — needed for the rogue-agent isolation demo (M3) and overhead measurement (M6).
- The **real semantic model** (M2) replaces the stub; `resolve_mu` loads it from `Mu`.
- **Real cryptographic signing** of `sigma` replaces the hash stub.

**After Milestone 1**, we proceed in order: real semantic model (M2), rogue-agent attack harness (M3), federated learning setup (M4), heterogeneity characterization (M5), overhead measurement (M6).

---

## What got dropped from the original repo

These were removed during the initial pruning pass and should not come back:
- `services/trust-scorer/` — central trust scoring is gone
- `services/pdp-behavior/` — central behavior PDP is gone
- `services/revocation-dispatcher/` — no central revocation
- `services/autoscaler/` — out of scope for this paper
- `services/auth/` — dynamic trust-evaluation role removed (may return as identity attestation)
- `zta-infrastructure/opa/` — no external policy
- `docker-compose.zta.yml`, `docker-compose.zta-grpc.yml`, `docker-compose.ablation.override.yml` — old topologies
- `agents-old/`, `test-driver-old/`, `random/`, `tests/ablation/` — legacy
- Old demo scripts and HTML

If anything in the active codebase references these, it's dead code and should be removed.

---

## What's kept and adapted

- All agents (`agents/supervisor`, `agents/travel-planner`, `agents/airline-agent`, `agents/hotel-agent`, `agents/car-rental-agent`)
- A2A framework and base agent (`agents/a2a/`, `agents/agent-base/`)
- MCP servers (`mcp-servers/airline`, `hotel`, `car-rental`)
- Backend services (`services/airline`, `services/hotel`, `services/car-rental`, `services/itinerary`)
- The sidecar (`zta-sidecar/`) — kept as the project but its middleware pipeline is being rewritten
- mTLS material (`zta-infrastructure/certs/`, `zta-infrastructure/envoy/`)
- `docker-compose.zta-mtls.yml` and `docker-compose.zta-sidecars.yml`
- `services/audit-logger/` — kept and adapted to collect FL training data
- `tests/`, `test-driver/`

---

## Things NOT to do

- **Don't keep the old trust scorer alongside the new model.** The whole point of the paper is that there is no central trust scorer. Don't try to make both coexist.
- **Don't add new agents or new domains.** The travel-planning testbed is enough. Adding more agent types is scope creep.
- **Don't optimize prematurely.** Milestone 1 uses stubs. Real models come in M2.
- **Don't write paper text.** The `.tex` skeleton stays mostly fixed until we have results to plug into tables. Paper writing happens after results exist, not concurrently.
- **Don't integrate any teammate's algorithm here.** A different project (Track 1) is working with a teammate's scoring algorithm. This project is self-contained.
- **Don't conflate this work with the original NeurIPS submission.** That work is a separate project in a separate repo. Different research question, different contribution. If something references "FedZTA" or the NeurIPS work, treat it as legacy and check whether it still applies.

---

## Working agreements

- **Ask before deleting anything substantial.** Pruning should be confirmatory, not destructive without confirmation.
- **Commit frequently.** Small focused commits per milestone. Easier to roll back.
- **Write tests as you go.** Each new component (message format, sidecar stage, attack class) gets at least one test.
- **Flag scope creep.** If a milestone is taking longer than expected, surface it.
- **Don't rewrite the agents themselves unless necessary.** Agents are the testbed, not the contribution.
- **Update this file** when milestones complete or when new decisions get made that should persist across sessions.

---

## Reference documents

These live alongside the codebase and should be consulted when context is needed:

- **`docs/ClaudeCode_Brief.docx`** — the full handoff brief; milestones, what-to-keep/drop, design notes
- **`docs/Track2_Brainstorm_v2.docx`** — design rationale, prior art positioning, open design questions
- **`docs/paper_skeleton.tex`** — paper section structure and algorithm pseudocode (the per-hop algorithm in §4.2 is the spec for the sidecar pipeline)
- **`docs/message_format.md`** — M1 spec for the message format (`content`, `mu`, `kappa`, `sigma`) and the per-hop pipeline, pointing to the code (`agents/a2a/trust.py`, `agents/a2a/substrate.py`)
- **`docs/abstract.md`** — the latest abstract version (also reproduced below for quick reference)

---

## Key open design questions

These will come up during implementation. They're not blockers for Milestone 1, but they need answers before Milestone 2 (real semantic model):

- What does "purpose" actually look like as data? A string label, a structured object, an embedding?
- Is `mu` one transformer or a bundle of separate small models with capability-specific heads?
- How does context get summarized when chains grow long?
- What does an agent's "attestation" contain? Identity, signed code hash, recent failure rate, declared purpose?
- How is `mu` trained initially? Synthetic data, supervised on the testbed corpus, distilled from a larger model?
- How is `sigma` maintained when `mu` and `kappa` change per hop? Resign with a per-hop key? Sign once and append signed deltas?

---

## Abstract (reference)

> Multi-agent AI systems today establish trust at the level of the principal: an agent is judged trustworthy or not based on its identity, history, and behavioral track record, and once judged trustworthy is given broad latitude over the messages it processes. This principal-centric model breaks down under the threat of rogue or compromised agents — a previously well-behaved agent that has been hijacked, substituted, or subverted still appears trustworthy to its callers, and there is no per-interaction signal to indicate otherwise. We propose a trust model in which trust is established per-message rather than per-principal, grounded in two evolving signals: the purpose the receiving agent declares for the interaction, and the context the message has accumulated over its journey through the agent chain. Each message carries a bundled semantic model describing what it is and what it has experienced so far; at every hop, this model — executing on a trusted compute substrate — evaluates whether the next agent's declared purpose is consistent with the message's intent and accumulated context, and only then permits the interaction to proceed. The same model also governs purpose-driven transformation of the message into a view appropriate for each agent, context-aware integration of agent responses as the chain progresses, and final synthesis of results back to the user from accumulated context rather than from any single agent. The semantic models that drive these decisions cannot be trained on centralized data: the training signal is generated locally across deployments, organizations, and agent networks, and contains exactly the kind of sensitive behavioral and contextual information the trust model is designed to protect. We therefore train the message-resident semantic models using federated learning, with each deployment contributing local updates from its own observed agent interactions and a shared trust model emerging without any deployment exposing raw data; the non-IID heterogeneity across deployments (different agent populations, task distributions, and threat profiles) presents a setting that existing federated learning methods have not been studied against, and we characterize how standard aggregation strategies behave on this class of data.
