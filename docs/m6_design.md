# Milestone 6 — Overhead & Ablation (design note)

M6 is the final milestone: it measures what the per-message trust model costs per
hop and compares it against the principal-based baseline (paper §6.3). Code:
`evaluation/overhead.py`, `evaluation/run_overhead.py`, `tests/test_overhead.py`.
(Builds on the M2 per-hop benchmark `benchmarks/trust_eval_latency.py`.)

## What it measures

Per-hop latency broken down by pipeline stage, timed in isolation (the stages are
pure given their inputs), so the numbers are the substrate's own overhead and
exclude the untrusted agent's execution:

    verify  ->  trust_eval  ->  transform  ->  integrate  ->  sign (resign)

for three configurations:
  - **stub** — the M1 stub model (floor),
  - **baseline** — principal-based trust (identity check, no purpose-scoped view),
  - **ours** — the learned per-message model.

Reports per-stage mean/p50/p95, total per-hop, and throughput (hops/s).

## Result (synthetic, `hashing:256` encoder)

| config   | verify | trust_eval | transform | integrate | sign | total | hops/s |
|----------|-------:|-----------:|----------:|----------:|-----:|------:|-------:|
| stub     | 0.006  | 0.000      | 0.000     | 0.002     | 0.006| 0.014 | ~70k   |
| baseline | 0.006  | 0.000      | 0.000     | 0.002     | 0.006| 0.014 | ~71k   |
| ours     | 0.006  | **0.059**  | 0.001     | 0.002     | 0.006| 0.074 | ~13.5k |

Findings:
- **`trust_eval` dominates** the trust model's overhead; the crypto stages
  (verify/sign) are ~0.006 ms each and transform/integrate are negligible.
- Ours adds ~0.06 ms/hop over the baseline with the hashing encoder.
- With the **real sentence-transformer encoder**, `trust_eval` is ~16 ms/hop
  (see `benchmarks/trust_eval_latency.py`) — the encoder forward pass is the cost
  driver, which is why M2 froze it and M4 federates only the head.

Run `python -m evaluation.run_overhead --encoder sentence-transformers:all-MiniLM-L6-v2`
for the real-encoder breakdown.

## Deferred (see TODO.md)
End-to-end throughput under the real over-the-wire substrate (needs the C#/
Python sidecar, not the in-process path); multi-core / batched trust_eval;
real-encoder committed numbers.
