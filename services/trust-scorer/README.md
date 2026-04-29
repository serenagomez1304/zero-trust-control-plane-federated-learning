# Trust Scorer Patch — Drop-In Changes

This folder contains **new and modified files** for adding the **trust scorer**
component to the Zero Trust Control Plane, as called for in the proposal:

> *"decomposed into separate services for PDP, auth gateway, **trust scorer**,
>  revocation dispatcher, and audit logger."*

## What's in the patch

### New files
```
services/trust-scorer/server.py          # FastAPI trust scoring service
services/trust-scorer/requirements.txt
services/trust-scorer/Dockerfile
smoke_test_scorer.py                     # 7 scenario tests (run before you commit)
```

### Modified files (overwrites in place)
```
zta-sidecar/Program.cs                   # adds trust-scorer middleware between MicroSeg and OPA
zta-infrastructure/opa/policy.rego       # adds trust_band check to allow rules
docker-compose.zta.yml                   # adds trust-scorer service + wires it into all 6 sidecars
```

## How to apply

From the root of your `zero-trust-control-plane/` repo:

```bash
# Assuming this patch is unpacked at /path/to/trust-scorer-patch/
cp -r /path/to/trust-scorer-patch/services/trust-scorer   services/
cp    /path/to/trust-scorer-patch/zta-sidecar/Program.cs  zta-sidecar/
cp    /path/to/trust-scorer-patch/zta-infrastructure/opa/policy.rego  zta-infrastructure/opa/
cp    /path/to/trust-scorer-patch/docker-compose.zta.yml  .

# Rebuild (sidecar needs rebuild because Program.cs changed)
docker compose -f docker-compose.zta.yml up --build

# Smoke test (scorer must be reachable on :8190)
python3 smoke_test_scorer.py
```

## How it works

**Middleware pipeline (sidecar):**
```
 1. Security Monitoring (log)
 2. WAF (SQLi/XSS/rate-limit)
 3. JWT Authentication
 4. Micro-Segmentation (ALLOWED_SOURCES)
 5. Trust Scorer   <-- NEW
 6. OPA Policy Engine (now receives trust_score + trust_band as input)
 7. DLP
 8. YARP reverse proxy
```

**Scoring model:**
```
T = 100 * ( 0.4*f_I + 0.4*f_B + 0.2*f_L )     score in [0, 100]

f_I  Identity strength     (JWT validity, sub-match, target-in-claim)
f_B  Behavioral history    (exp-decayed penalties from past violations)
f_L  LLM-layer signal      (prompt-injection pattern detection on A2A payload)
```

**Threshold bands:**
| Score | Band | Action |
|------:|------|--------|
| ≥ 80  | allow   | forward to OPA |
| 60–79 | step_up | forward to OPA with `trust_band=step_up` (logged) |
| < 60  | block   | 403 at sidecar, never reaches OPA or upstream |

**Hard-floor rules** (block regardless of composite score):
- Identity completely fails (no token, expired, or sub-mismatch) → **block**
- Prompt-injection mass ≥ 0.7 on request payload → **block** + record in history

## Citations (inline in the code)

| Design decision | Citation |
|---|---|
| Weighted-sum trust model | Dimitrakos et al., IEEE TrustCom 2020, DOI 10.1109/TrustCom50675.2020.00247 |
| Threshold bands (80/60) | Kim & Lee, *Applied Sciences* 15(17):9551, 2025 (MDPI) |
| Per-path dynamic thresholds | Bicakci, Schmid & Settanni, arXiv:2402.08299, 2024 |
| Exponential decay on history | Sun et al., *IEEE JSAC* 43(6):2089, June 2025 |
| LLM-layer signal | He et al., arXiv:2506.02546, 2025 |
| ZTA-for-multi-LLM framing | Liu et al., arXiv:2508.19870, 2025 |
| Trust-score authz as canonical ZTA control | Poirrier et al., *Proc. IEEE* 113(1), Jan 2025 |

## Key endpoints

Once running on port 8190:

```bash
# Health + self-documenting citations
curl http://localhost:8190/health

# Score a request
curl -X POST http://localhost:8190/score -H 'Content-Type: application/json' -d '{
  "agent_id": "supervisor-agent",
  "target_component": "airline-agent-sidecar",
  "path": "/a2a",
  "method": "POST",
  "jwt_token": "<JWT>",
  "payload_text": "Find flights JFK to LAX"
}'

# Record a behavioral event (called by DLP/OPA/WAF on violations)
curl -X POST http://localhost:8190/feedback -H 'Content-Type: application/json' -d '{
  "agent_id": "hotel-agent",
  "severity": 0.7,
  "reason": "dlp-pii-leak"
}'

# Inspect an agent's history
curl http://localhost:8190/agents/hotel-agent/history
```

## What's NOT wired yet (intentional)

- **Feedback loop from sidecars**: currently the sidecar only *reads* the trust
  score. The `/feedback` endpoint exists but nothing calls it automatically.
  Next step: have the OPA-deny, WAF-block, and DLP-hit middleware POST to
  `/feedback` with an appropriate severity. This turns each observed violation
  into future trust degradation — matching the "continuous evaluation" principle
  of Dimitrakos et al. 2020.

- **Token replay detection**: JWTs are validated but not tracked. Adding a jti
  (JWT ID) replay cache in the scorer would let you cite the "token replay
  storms" adversarial test explicitly mentioned in your proposal.

- **Prometheus metrics**: useful for the "benchmark latency with/without trust
  scorer" experiment, but not required for correctness.

## Tested scenarios (all pass)

```
S1 clean supervisor, benign payload                  -> score 100, allow
S2 NO JWT (identity failure on gated path)           -> score  60, block  [hard-floor]
S3 prompt injection in A2A payload                   -> score  80, block  [hard-floor]
S4 agent with recent policy violations               -> score  60, step_up
S5 same agent, 2 hours later (decay recovers trust)  -> score  85, allow
S6 impersonation (token sub != declared agent_id)    -> score  60, block  [hard-floor]
S7 health path never gated                           -> allow regardless
```
