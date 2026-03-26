# Demo Script & Screenshot Guide

Run these commands in order. After each section, take a screenshot of your terminal output.
Name your screenshots as indicated — they'll go into the report.

## Prerequisites
```bash
cd /Users/serenagomez/zero-trust-control-plane
docker compose -f docker-compose.zta.yml up --build
# Wait for all containers to be healthy (~30-60 seconds)
```

---

## Screenshot 1: System Health (all services up)
```bash
echo "=== Backend Services ==="
curl -s http://localhost:8001/health | python3 -m json.tool
curl -s http://localhost:8002/health | python3 -m json.tool
curl -s http://localhost:8003/health | python3 -m json.tool

echo "=== MCP Servers (SSE) ==="
curl -sI http://localhost:8010/sse 2>&1 | head -5
curl -sI http://localhost:8011/sse 2>&1 | head -5
curl -sI http://localhost:8012/sse 2>&1 | head -5

echo "=== Agents ==="
curl -s http://localhost:8091/health | python3 -m json.tool
curl -s http://localhost:8092/health | python3 -m json.tool
curl -s http://localhost:8093/health | python3 -m json.tool

echo "=== Supervisor ==="
curl -s http://localhost:8080/health | python3 -m json.tool

echo "=== ZTA Services ==="
curl -s http://localhost:8181/health | python3 -m json.tool
curl -s http://localhost:8180/health | python3 -m json.tool
```
**Save as**: `screenshot-01-system-health.png`

---

## Screenshot 2: A2A Agent Card Discovery
```bash
curl -s http://localhost:8091/.well-known/agent.json | python3 -m json.tool
```
**Save as**: `screenshot-02-agent-card.png`

---

## Screenshot 3: Supervisor Agent Discovery (through sidecars)
```bash
curl -s http://localhost:8080/agents | python3 -m json.tool
```
**Save as**: `screenshot-03-supervisor-agents.png`

---

## Screenshot 4: JWT Token Issuance & Verification
```bash
# Issue token
curl -s -X POST http://localhost:8180/token \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "supervisor-agent", "secret": "zta-agent-shared-secret"}' | python3 -m json.tool

# Verify token
TOKEN=$(curl -s -X POST http://localhost:8180/token -H "Content-Type: application/json" \
  -d '{"agent_id": "supervisor-agent", "secret": "zta-agent-shared-secret"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -s -X POST http://localhost:8180/token/verify \
  -H "Content-Type: application/json" \
  -d "{\"token\": \"$TOKEN\"}" | python3 -m json.tool
```
**Save as**: `screenshot-04-jwt-auth.png`

---

## Screenshot 5: Supervisor JWT Acquisition (from logs)
```bash
docker compose -f docker-compose.zta.yml logs supervisor 2>&1 | grep -i "jwt\|token" | head -10
```
**Save as**: `screenshot-05-jwt-acquisition.png`

---

## Screenshot 6: End-to-End Flight Search
```bash
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find flights from JFK to LAX on 2025-07-15"}' | python3 -m json.tool
```
**Save as**: `screenshot-06-flight-search.png`

---

## Screenshot 7: End-to-End Hotel Search
```bash
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Search for hotels in New York from July 15 to July 18"}' | python3 -m json.tool
```
**Save as**: `screenshot-07-hotel-search.png`

---

## Screenshot 8: End-to-End Car Rental Search
```bash
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find rental cars at LAX from July 15 to July 20"}' | python3 -m json.tool
```
**Save as**: `screenshot-08-car-rental-search.png`

---

## Screenshot 9: ZTA Sidecar Health
```bash
curl -s http://localhost:19091/sidecar/health | python3 -m json.tool
```
**Save as**: `screenshot-09-sidecar-health.png`

---

## Screenshot 10: ZTA Enforcement — Authorized vs Denied
```bash
echo "=== ALLOWED: supervisor-agent ==="
curl -s -H "x-agent-id: supervisor-agent" http://localhost:19091/.well-known/agent.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Status: OK — Agent: {d[\"name\"]}')"

echo ""
echo "=== DENIED: evil-agent ==="
curl -s -H "x-agent-id: evil-agent" http://localhost:19091/a2a | python3 -m json.tool

echo ""
echo "=== DENIED: no identity ==="
curl -s http://localhost:19091/a2a | python3 -m json.tool
```
**Save as**: `screenshot-10-zta-enforcement.png`

---

## Screenshot 11: OPA Policy Decisions
```bash
echo "=== ALLOW: supervisor → airline-agent ==="
curl -s -X POST http://localhost:8181/v1/data/zta/authz/allow \
  -H "Content-Type: application/json" \
  -d '{"input": {"agent_id": "supervisor-agent", "path": "/a2a", "method": "POST", "component": "airline-agent-sidecar"}}' | python3 -m json.tool

echo ""
echo "=== DENY: evil-agent → airline-agent ==="
curl -s -X POST http://localhost:8181/v1/data/zta/authz/allow \
  -H "Content-Type: application/json" \
  -d '{"input": {"agent_id": "evil-agent", "path": "/a2a", "method": "POST", "component": "airline-agent-sidecar"}}' | python3 -m json.tool
```
**Save as**: `screenshot-11-opa-policy.png`

---

## Screenshot 12: ZTA Response Headers
```bash
curl -sI -H "x-agent-id: supervisor-agent" http://localhost:19091/health 2>&1 | grep -i "x-zta\|HTTP"
```
**Save as**: `screenshot-12-zta-headers.png`

---

## Screenshot 13: A2A Task Through Sidecar (full JSON-RPC round-trip)
```bash
curl -s -X POST http://localhost:19091/a2a \
  -H "Content-Type: application/json" \
  -H "x-agent-id: supervisor-agent" \
  -d '{
    "jsonrpc": "2.0",
    "id": "demo-1",
    "method": "tasks/send",
    "params": {
      "id": "demo-task-001",
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "List all supported airports"}]
      }
    }
  }' | python3 -m json.tool
```
**Save as**: `screenshot-13-a2a-task-sidecar.png`

---

## Screenshot 14: Benchmark Results
```bash
source .venv/bin/activate
python benchmarks/latency_benchmark.py --runs 10 --suite all
```
**Save as**: `screenshot-14-benchmarks.png`

---

## Screenshot 15: Autoscaler Dry Run
```bash
python services/autoscaler/autoscaler.py --once --dry-run
```
**Save as**: `screenshot-15-autoscaler-dry-run.png`

---

## Screenshot 16: Autoscaler Live Scaling
```bash
python services/autoscaler/autoscaler.py --once
```
**Save as**: `screenshot-16-autoscaler-live.png`

---

## Screenshot 17: Docker Containers Running
```bash
docker compose -f docker-compose.zta.yml ps
```
**Save as**: `screenshot-17-containers.png`

---

## Screenshot 18: Unit Tests Passing
```bash
source .venv/bin/activate
pytest tests/test_a2a.py -v
```
**Save as**: `screenshot-18-unit-tests.png`