# Milestone 5 — Non-IID Heterogeneity Characterization (design note)

M5 answers the paper's §5.3 question: *how heterogeneous are the deployments,
and does that heterogeneity explain FL performance?* Code:
`agents/a2a/federated/heterogeneity.py`, `evaluation/run_heterogeneity.py`,
`tests/test_heterogeneity.py`.

## What it measures

For each of the paper's three heterogeneity sources we form each client's
categorical marginal and measure its **Jensen-Shannon divergence** (symmetric,
in [0, 1], log base 2) from the pooled marginal; the partition's heterogeneity on
that source is the mean over clients (KL is also available):

| source            | marginal over            |
|-------------------|--------------------------|
| agent_population  | declared purpose (`purpose_label`) |
| task_distribution | task domain of the intent (keyword-inferred) |
| threat_profile    | `kind` = legit / cross_domain / escalation |

`overall` is the mean of the three.

## Sweep and correlation

`run_heterogeneity.py` sweeps the Dirichlet α (IID → α=0.05), and at each level
reports the heterogeneity scores and the FL accuracy of all four strategies,
then the Pearson correlation between overall heterogeneity and accuracy.

Result (synthetic, `hashing:256`, K=5):
- Heterogeneity rises smoothly as α falls (overall ≈0.004 at IID → ≈0.167 at α=0.05).
- FL accuracy falls as heterogeneity rises, with a **strong negative
  correlation** for every strategy (r ≈ −0.84 to −0.95). SCAFFOLD degrades most
  under extreme skew.

## Honest limitation

The M4 partition keys on `(label, kind)`, so it induces **agent-population** and
**threat-profile** skew but leaves **task_distribution** heterogeneity near zero
(each client sees ~the same task-domain mix). That is expected given the
partition key, and the characterization reports it faithfully rather than hiding
it. Inducing task heterogeneity would mean adding the task domain to the
partition key — a knob left for future work (noted in TODO.md).

## Deferred (see TODO.md)
Task-heterogeneity partitioning; Wasserstein on an ordered/embedded space
(JS/KL suffice for these categorical marginals); larger K and multi-seed
variance bands.
