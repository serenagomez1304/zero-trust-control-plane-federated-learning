# Per-Stage Ablation Patch

Tier 1 / Item 1 from the Stanford reviewer's actionable list:

> **Q4. Can you provide an ablation showing the marginal contribution of
> each pipeline stage (WAF, revocation, micro-segmentation, trust scorer,
> behavior PDP, OPA) to overall defense success?**

This patch produces exactly that table.

## What it does

For every scenario in `test-driver/prompts/transit_trust.json` and
`test-driver/prompts/prompt_injection.json`, the harness runs the request
through the airline-agent sidecar in **7 configurations**:

| Configuration | Stages active |
|---|---|
| `BASELINE`    | All stages |
| `NO_WAF`      | All except WAF |
| `NO_REVOC`    | All except revocation dispatcher check |
| `NO_MICROSEG` | All except micro-segmentation |
| `NO_TRUST`    | All except trust scorer |
| `NO_BEHAVIOR` | All except behavior PDP |
| `NO_OPA`      | All except authorization PDP |

For each (scenario, configuration) pair it records HTTP status, the stage
that denied the request (resolved from `X-ZTA-*` response headers and
body text), and request latency.

Outputs a four-file artifact set:

  - `raw_trials.csv`  — one row per (scenario, configuration); appendix-grade
  - `pivot.csv`       — scenarios × configurations, cell = denying stage
  - `pivot.md`        — same as pivot.csv, paste-ready into the paper
  - `summary.txt`     — headline numbers (per-config deny rate, marginal
                        contribution per stage, baseline stage attribution)

## What this patch changes in the codebase

| File | Change |
|---|---|
| `zta-sidecar/Program.cs` | Two new env flags — `ENABLE_MICROSEG` and `ENABLE_OPA` — both defaulting to enabled. Without these, ablating the micro-seg and OPA stages would require code edits per run. |
| `tests/ablation/run_ablation.py` | NEW. The harness. |

The sidecar change is intentionally minimal. Existing deployments are
unaffected — only the ablation harness ever sets these flags to `false`.

## Apply

From the project root:

```bash
# 1. Patched sidecar (rebuild required)
cp ablation-patch/zta-sidecar/Program.cs zta-sidecar/Program.cs

# 2. Harness
mkdir -p tests/ablation
cp ablation-patch/tests/ablation/run_ablation.py tests/ablation/

# 3. Rebuild the airline-agent sidecar (the only one the harness mutates)
docker compose -f docker-compose.zta.yml up -d --build airline-agent-sidecar
```

## Run

The full testbed must be up and healthy first
(`python tests/test_federated_loop.py --debug` should pass before running
the ablation).

```bash
# Smoke run — first 5 scenarios, BASELINE only, no docker reconfig
python tests/ablation/run_ablation.py --no-docker --max-scenarios 5

# Full run — 32 scenarios × 7 configurations = 224 trials
# Takes ~5–10 minutes (most of which is sidecar recreation between configs)
python tests/ablation/run_ablation.py
```

Results go to `tests/ablation/results/<UTC-timestamp>/`.

## Reading the output

`summary.txt` has the headline numbers your paper needs. The most useful
line is the **marginal contribution per stage** — for each ablated stage,
how many scenarios that BASELINE denied are now ALLOWED. If a stage shows
0 marginal contribution, the paper's claim that every component matters
is in trouble; if every stage shows non-zero contribution, you have the
empirical answer to Q4.

`pivot.md` is what goes in the paper. Each row is one attack scenario,
each column is one configuration, each cell tells you which stage caught
the attack (or `ALLOWED` if it leaked through). Reviewers can scan this
to see exactly which stage handled which class of attack.

## Caveats worth noting in the paper

  - **Single sidecar.** The harness only mutates `airline-agent-sidecar`,
    so the ablation measures the contribution of each stage *as observed
    at one PEP*. The federated story (revocation propagating to all
    sidecars) is verified separately by `test_federated_loop.py`.

  - **JWT and DLP are not ablated.** JWT removal would invalidate the
    identity model that every other stage depends on (it's not really
    a "stage" you can remove in isolation). DLP runs on the response
    side and doesn't contribute to deny decisions on the request path,
    so its ablation would be a no-op for this measurement.

  - **Stage order matters.** A WAF block on a scenario that *would also*
    have triggered the trust scorer is attributed to WAF — because that's
    the order the pipeline runs. Reviewers asking "which stages would
    have caught this if WAF were removed?" are answered by the `NO_WAF`
    configuration column, which is exactly what we want.

  - **32 scenarios, not 49.** The paper reports 49; the on-disk corpus
    is 6 transit + 26 prompt-injection = 32. The remaining 17 are load
    and intent scenarios that don't have a single deny/allow ground
    truth and aren't appropriate for an ablation. State this in the
    paper or backfill the corpus before publishing.
