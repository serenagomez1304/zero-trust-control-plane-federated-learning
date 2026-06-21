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

**Milestone 1: end-to-end message flow with stubs.**

Goal: get one message moving through the system with the new structure (`content`, `mu`, `kappa`, `sigma`), the new 4-stage sidecar pipeline (verify → trust_eval → transform → integrate), and an end-to-end demo. The semantic model is stubbed at this stage:
- `mu.trust_eval` always returns 1.0
- `mu.transform` returns content unchanged
- `mu.integrate` appends response to context
- `mu.synthesize` returns the latest response

Acceptance criteria:
- Single user request flows end-to-end through one agent chain (user → supervisor → airline-agent)
- Message format is documented and the schema is in code
- Sidecar middleware pipeline executes the 4 stages
- All previous trust-scorer references are removed from the active code path
- Tests in `tests/test_a2a.py` pass with the new message format

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
