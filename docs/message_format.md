# Message Format & Per-Hop Pipeline (Milestone 1)

This document specifies the per-message trust model's wire structure and the
substrate pipeline that operates on it. The authoritative schema lives in code:

- **`agents/a2a/trust.py`** — message format and the semantic model.
- **`agents/a2a/substrate.py`** — the trusted compute substrate and per-hop pipeline.
- **`demo/trust_chain_demo.py`** — runnable `user → supervisor → airline-agent` demo.
- **`tests/test_a2a.py`** — tests for all of the above.

It corresponds to §4 of `docs/paper_skeleton.tex`.

> **Milestone 1 status: stubbed.** The *structure* and the *pipeline* are real;
> the *semantic model* (`mu`) is a stub. The trained model arrives in M2.

---

## Message

A message is the 4-tuple

```
m = ⟨ content, mu, kappa, sigma ⟩
```

| Field     | Symbol | Type (`trust.py`) | Meaning |
|-----------|--------|-------------------|---------|
| `content` | —      | `Content`         | The payload. Never mutated in transit; agents see only a *view* of it. |
| `mu`      | μ      | `Mu`              | Descriptor of the bundled semantic model that governs trust decisions. |
| `kappa`   | κ      | `Kappa`           | Accumulated context of the message's journey. Starts with the user intent and no entries. |
| `sigma`   | σ      | `Sigma`           | Signature chain authenticating `⟨content, mu, kappa⟩`; one signature appended per hop. |

### `Content`
```
payload : str              # primary payload (e.g. the user request)
data    : dict             # optional structured payload
```

### `Mu` (semantic model descriptor)
```
model_id : str             # e.g. "stub-semantic-model"
version  : str             # e.g. "0.1.0-stub"
bundle   : list[str]       # ["trust_eval", "transform", "integrate", "synthesize"]
params   : dict            # opaque model parameters/refs (empty for the M1 stub)
```
`Mu` is the *data* that travels in the message. The *behavior* is a
`SemanticModel` resolved from it by `resolve_mu(mu)` — in M1 always the
`StubSemanticModel`; in M2 the real model identified/parameterized by `Mu`.

### `Kappa` (accumulated context)
```
intent  : str                      # the originating user intent
entries : list[ContextEntry]       # one per hop, in chain order
                                   #   ContextEntry = { purpose: str, response: str }
```

### `Sigma` (signature chain)
```
chain : list[str]          # signatures; one per hop (origin + each accepted hop)
```
M1 signatures are keyless `sha256` digests over the canonical JSON of
`⟨content, mu, kappa⟩`, prefixed `stub-sig:`. This makes `verify` tamper-detecting
but is **not** authentication — real per-hop signing (keys / attester) is deferred.

### Per-agent inputs to trust evaluation
Not part of the message; supplied by each hop's target agent.
```
DeclaredPurpose = { label: str, description: str }          # p_a
Attestation     = { agent_id, code_hash?, signed_by?, recent_failure_rate? }   # alpha_a
```

---

## The semantic model `mu`

Four capabilities (`SemanticModel` in `trust.py`), executed by the substrate —
never by agents. M1 stub behavior (`StubSemanticModel`):

| Capability | Signature | M1 stub behavior |
|------------|-----------|------------------|
| `trust_eval` | `(purpose, attestation, kappa) → float` | always returns **1.0** |
| `transform`  | `(content, purpose, kappa) → Content`   | returns content **unchanged** |
| `integrate`  | `(response, kappa, purpose) → Kappa`    | **appends** the response to context |
| `synthesize` | `(kappa) → str`                          | returns the **latest** response |

---

## Per-hop pipeline (`process_hop`)

At each hop the substrate runs (Algorithm 1, §4.2):

```
1. verify       verify_signature(m) over ⟨content, mu, kappa⟩   — reject if invalid
2. trust_eval   t = mu.trust_eval(p_a, alpha_a, kappa)          — reject if t < tau
3. transform    view = mu.transform(content, p_a, kappa)        — the only thing the agent sees
4. agent runs   r = agent(view)                                  — untrusted
5. integrate    kappa' = mu.integrate(r, kappa, p_a)
                resign:  sigma' = sigma + [sign(content, mu, kappa')]
                m' = ⟨content, mu, kappa', sigma'⟩
```

`tau` is `TRUST_THRESHOLD = 0.5`. Since the stub `trust_eval` returns 1.0, the
accept path runs by default; the reject path is reachable with a higher
threshold or a non-stub model.

**Security-relevant invariants** (the source of compromise-bounding):
- The agent is invoked only on its `view`; `content` is never mutated or handed
  to an agent directly.
- The agent's output enters the message only through `integrate`.
- The agent never produces the user-facing output — `synthesize` does.

## End of chain (`synthesize`)

```
response = mu.synthesize(kappa)
```
The user-facing response is built on the trusted substrate from accumulated
context, not by any single agent.

## Driver (`run_chain`)

`run_chain(message, agents, threshold)` runs `process_hop` for each agent in
order, stopping at the first rejected hop (`completed = False`); on success it
synthesizes and returns the final response. Returns a `ChainResult` with the
final message, the synthesized response, and a per-hop `HopResult` list.

---

## Deferred to later milestones

These are intentionally out of scope for M1 (in-process, stubbed) and are
tracked as future work:

- **Real semantic model (M2).** Replace `StubSemanticModel`; `resolve_mu` loads
  the model described by `Mu`.
- **Over-the-wire substrate (M3, M6).** Run the pipeline in the real substrate —
  the C# `zta-sidecar` (or a Python sidecar process) fronting each agent — with
  the trust envelope propagated in A2A message metadata. Needed to demonstrate
  the isolation boundary against rogue agents (M3) and to measure overhead (M6).
- **Real signing of `sigma`.** Replace the keyless hash stub with per-hop
  cryptographic signatures (keys / trusted attester).
