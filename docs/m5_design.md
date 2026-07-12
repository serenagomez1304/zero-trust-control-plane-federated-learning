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
| task_distribution | task domain of the intent (`TrustExample.domain`, recorded at generation) |
| threat_profile    | `kind` = legit / cross_domain / escalation |

`overall` is the mean of the three.

## Sweep and correlation

`run_heterogeneity.py` sweeps the Dirichlet α (IID → α=0.05), and at each level
reports the heterogeneity scores and the FL accuracy of all four strategies,
then the Pearson correlation between overall heterogeneity and accuracy.

Result (synthetic, `hashing:256`, K=5, seed-averaged over 3 seeds):
- Heterogeneity rises smoothly as α falls (overall ≈0.003 at IID → ≈0.245 at
  α=0.05), and **all three sources move** — at α=0.05 agent-population ≈0.30,
  task-distribution ≈0.26, threat-profile ≈0.17.
- FL accuracy (mean±std over seeds) falls as heterogeneity rises, with a
  **strong negative correlation** for every strategy (r ≈ −0.84 to −0.999), and
  the accuracy variance grows under stronger skew.

The sweep is averaged over seeds (`--seeds`, default 3): for each α, both the
heterogeneity scores and each strategy's accuracy are averaged over seeds (the
seed varies the partition and the FL run), and the correlation is computed on
the seed-means.

## All three sources exercised

The partition key is `(label, kind, domain)`, where `domain` is the intent's task
domain recorded on each `TrustExample` at generation. This induces skew in all
three sources the paper names — agent-population (`purpose_label`),
task-distribution (`domain`), and threat-profile (`kind`) — so §5.3 characterizes
the full non-IID setting rather than two of three. (Earlier the key was
`(label, kind)`, which left task-distribution heterogeneity ~0.)

## Deferred (see TODO.md)
Wasserstein on an ordered/embedded space
(JS/KL suffice for these categorical marginals); larger K and multi-seed
variance bands.
