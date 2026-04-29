# Workload Identity Patch (Option B)

Adds SPIFFE-style workload identity so every agent acquires its own JWT at
startup and propagates it on outbound calls. This fixes the `trust_score:0`
issue you saw on Phase 1 of the federated-loop test, where the trust scorer
was returning `certain_false` identity for every request because no token
was being presented.

## What this changes

| File | Change |
|---|---|
| `agents/a2a/auth.py` | **NEW.** `WorkloadIdentity` class — JWT bootstrap, TTL-aware refresh, fail-safe headers |
| `agents/airline-agent/agent.py` | Bootstraps `WorkloadIdentity` at startup, passes Bearer token to MCP via `headers={...}` in MCP config |
| `agents/hotel-agent/agent.py` | Same patch as airline |
| `agents/car-rental-agent/agent.py` | Same patch as airline |
| `services/trust-scorer/server.py` | Emits `trust.injection` events to audit logger when `injection_mass >= 0.7` |
| `services/trust-scorer/requirements.txt` | Adds `httpx==0.27.2` |
| `docker-compose.zta.yml` | Adds `ZTA_AUTH_URL` + `ZTA_AGENT_SECRET` to the three worker agents; adds `AUDIT_LOGGER_URL` + `depends_on: audit-logger` to trust-scorer |
| `tests/test_federated_loop.py` | Phase 1 + Phase 2 acquire real JWT for `supervisor-agent` from auth service; Phase 1 now asserts `trust_score >= 60` |

## Architecture (SPIFFE delegation pattern)

Inbound A2A request → sidecar validates the **caller's** token.
Outbound MCP call → agent presents **its own** token (acquired at startup).

The micro-segmentation rule on the MCP sidecar is `ALLOWED_SOURCES=airline-agent`,
not `supervisor-agent`. That's why the agent must use its own identity
on the outbound call — the supervisor's identity wouldn't pass micro-seg
even if we replayed it.

## Apply

From the project root:

```bash
# 1. New helper module
mkdir -p agents/a2a
cp workload-identity-patch/agents/a2a/auth.py agents/a2a/auth.py

# 2. Patched worker agents
cp workload-identity-patch/agents/airline-agent/agent.py     agents/airline-agent/agent.py
cp workload-identity-patch/agents/hotel-agent/agent.py       agents/hotel-agent/agent.py
cp workload-identity-patch/agents/car-rental-agent/agent.py  agents/car-rental-agent/agent.py

# 3. Patched trust scorer
cp workload-identity-patch/services/trust-scorer/server.py        services/trust-scorer/server.py
cp workload-identity-patch/services/trust-scorer/requirements.txt services/trust-scorer/requirements.txt

# 4. Patched compose
cp workload-identity-patch/docker-compose.zta.yml docker-compose.zta.yml

# 5. Patched test
cp workload-identity-patch/tests/test_federated_loop.py tests/test_federated_loop.py

# 6. Rebuild and run
docker compose -f docker-compose.zta.yml down
docker compose -f docker-compose.zta.yml up --build
```

## Verify

In a second terminal once everything is healthy:

```bash
# Should show ALL FIVE phases pass — including the new
# "trust score healthy: score=85+ band=allow" line in Phase 1
python tests/test_federated_loop.py --debug
```

You can also confirm the agents actually got their JWTs at startup:

```bash
docker compose -f docker-compose.zta.yml logs airline-agent | grep "workload identity"
# expected: workload identity acquired for airline-agent
```

And see the trust-scorer's injection events landing in audit:

```bash
curl -s 'http://localhost:8195/events?event_type=trust.injection' | python3 -m json.tool
# expected: source: "trust-scorer" entries with hits + injection_mass
```

## What this enables for the report

This is the piece that lets you say "the system implements SPIFFE-style
workload identity" with a straight face. Concretely:

  1. **Every hop is authenticated.** Supervisor → airline-agent (caller's JWT),
     airline-agent → airline-mcp (agent's own JWT). Both are validated by
     the destination sidecar before any policy evaluation.

  2. **The trust scorer's identity opinion now produces useful signal.**
     With JWTs flowing, `b_I` and `u_I` are non-trivial functions of the
     token's freshness, signature validity, sub/agent_id match, and target
     in `allowed_targets`. Before this patch every projection collapsed to 0
     and the SL aggregator was effectively dominated by a single sink.

  3. **The audit log is now causally complete.** The behavior PDP's R2 rule
     fires on N injection events in a window — and those events now arrive
     from the actual production trust scorer, not from test seeds. You can
     point at a real `trust.injection → behavior.deny → revocation.issued`
     chain in the audit timeline as evidence the federated control plane
     is operating end-to-end.

## Caveats worth flagging in the report

  - **Shared secret bootstrap.** Each worker agent has the same hardcoded
    `ZTA_AGENT_SECRET` in compose. In production this would be per-agent,
    rotated, and provisioned via a workload attestation flow (SPIRE node
    attestation, K8s service account tokens, etc.). For this study the
    threat model boundary is "the docker-compose host is trusted."

  - **JWT lifetime: 24h.** No refresh-after-401 wired through the MCP
    SSE channel (the helper supports `force_refresh` but the agent
    doesn't call it on a 401 from MCP). Acceptable for a testbed where
    runs are short; a real deployment would refresh proactively.

  - **No mTLS underneath.** Tokens are bearer tokens over HTTP. A real
    deployment would terminate mTLS at the sidecar. This is consistent
    with the rest of the testbed's threat model.
