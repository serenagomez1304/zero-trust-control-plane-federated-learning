# Milestone 3 — Rogue-Agent Attack Harness (design note)

M3 is the security-evaluation milestone. It implements the three attack classes
from the threat model (paper §3.3) and produces the Table 1 data (§6.1). See the
code: `evaluation/attacks.py`, `evaluation/run_attacks.py`, `tests/test_attacks.py`.

## What it measures

Each attack scenario runs through two configurations:

- **baseline** — principal-based trust (`PrincipalBaselineModel`): trust any
  *known* agent with clean history, ignoring the declared purpose and context,
  and with no purpose-scoped view (broad latitude once trusted).
- **ours** — the per-message model: learned `trust_eval` on declared purpose vs.
  intent/context, a purpose-scoped `transform`, and the attestation gate.

Metrics per (class, config): **ASR** (attack success rate — fraction of
scenarios where the malicious hop was accepted) and **exposure** (mean fraction
of the message's sensitive fields visible in the malicious agent's view).

## Attack classes and the mechanism that catches each

| class            | attacker does                                   | caught by (ours) |
|------------------|-------------------------------------------------|------------------|
| substitution     | swap in a malicious agent (same credentials)    | `trust_eval`     |
|                  | that declares a rogue purpose                    | (purpose)        |
| prompt_injection | legit agent hijacked into a rogue purpose        | `trust_eval`     |
| supply_chain     | agent code/model tampered; purpose still legit   | attestation gate |

Injection scenarios are derived from `test-driver/prompts/prompt_injection.json`
(26 prompts across 15 categories), mapping each category to the rogue purpose the
injection induces.

## Modeling choices (stated for honesty)

- **One mechanism per class.** To attribute each result cleanly, substitution and
  injection agents present a *valid* attestation but a *rogue* purpose (so
  `trust_eval` is the detector), while the supply-chain agent presents a
  *legitimate* purpose but an *invalid* attestation (so the attestation gate is
  the detector). `tests/test_attacks.py` verifies this attribution by disabling
  the other defense. In reality a substituted agent might also fail attestation;
  the split is for clean measurement, and both defenses are always active in
  "ours".
- **Synthetic, controlled setting.** With a `trust_eval` at ~100% accuracy on the
  synthetic data and a deterministic attestation gate, the result is a clean
  baseline 1.00 / ours 0.00 on every class. This demonstrates the *mechanism*;
  it is not a claim about real-world detection rates. More realistic difficulty
  (unseen rogue purposes, noisier labels) is future work.
- **Rogue purposes are drawn from the known escalation set** the model was
  trained against, so detection does not depend on generalization to unseen
  purposes. Testing that generalization is a separate robustness question
  (tracked in TODO.md).

## Deferred
Context-integration drift as a secondary rogue signal; the over-the-wire
isolation demo (needs the real substrate); `trust_eval` generalization to
unseen rogue purposes. See TODO.md.
