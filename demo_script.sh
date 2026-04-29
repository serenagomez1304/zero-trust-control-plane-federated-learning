#!/bin/bash
# =============================================================================
# ZTA Multi-Agent System — Full Demo Script
# =============================================================================
# Runs all ZTA enforcement demos and captures output for screen recording.
# Usage: bash demo_script.sh [--quick]
#   --quick: skip load tests (faster demo)
# =============================================================================

set -e

SUPERVISOR_URL="http://localhost:8080"
AIRLINE_SIDECAR="http://localhost:19091"
HOTEL_SIDECAR="http://localhost:19092"
CARRENTAL_SIDECAR="http://localhost:19093"
AIRLINE_MCP_SIDECAR="http://localhost:19010"
OPA_URL="http://localhost:8181"
AUTH_URL="http://localhost:8180"
AIRLINE_DIRECT="http://localhost:8091"

QUICK=false
[[ "$1" == "--quick" ]] && QUICK=true

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

divider() { echo -e "${DIM}$(printf '─%.0s' {1..72})${NC}"; }
header() { 
  echo ""
  echo -e "${CYAN}$(printf '═%.0s' {1..72})${NC}"
  echo -e "${CYAN}${BOLD}  $1${NC}"
  echo -e "${CYAN}$(printf '═%.0s' {1..72})${NC}"
  echo ""
}
pass() { echo -e "  ${GREEN}✓ PASS${NC} — $1"; }
fail() { echo -e "  ${RED}✗ FAIL${NC} — $1"; }
info() { echo -e "  ${YELLOW}→${NC} $1"; }

# =============================================================================
header "ZTA MULTI-AGENT SYSTEM — FULL DEMO"
echo -e "  Date: $(date)"
echo -e "  System: 18 containers, 6 ZTA sidecars, OPA + JWT Auth"
echo ""

# =============================================================================
header "1. SYSTEM HEALTH CHECK"
# =============================================================================

info "Checking supervisor..."
curl -sf ${SUPERVISOR_URL}/health | python3 -m json.tool && pass "Supervisor healthy" || fail "Supervisor down"
divider

info "Checking agent sidecars..."
for svc in "airline-agent-sidecar:${AIRLINE_SIDECAR}" "hotel-agent-sidecar:${HOTEL_SIDECAR}" "car-rental-agent-sidecar:${CARRENTAL_SIDECAR}"; do
  name="${svc%%:*}"
  url="${svc##*:}"
  curl -sf ${url}/sidecar/health | python3 -m json.tool && pass "${name}" || fail "${name}"
done
divider

info "Checking OPA..."
curl -sf ${OPA_URL}/health | python3 -m json.tool && pass "OPA healthy" || fail "OPA down"
divider

info "Checking JWT Auth Service..."
curl -sf ${AUTH_URL}/health | python3 -m json.tool && pass "Auth Service healthy" || fail "Auth Service down"

# =============================================================================
header "2. END-TO-END AGENT QUERIES"
# =============================================================================

info "Flight search..."
curl -s -X POST ${SUPERVISOR_URL}/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find flights from JFK to LAX on 2025-07-15"}' | python3 -m json.tool
divider

info "Hotel search..."
curl -s -X POST ${SUPERVISOR_URL}/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Search for hotels in New York from July 15 to July 18"}' | python3 -m json.tool
divider

info "Car rental search..."
curl -s -X POST ${SUPERVISOR_URL}/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find rental cars at LAX from July 15 to July 20"}' | python3 -m json.tool

# =============================================================================
header "3. ZTA MICRO-SEGMENTATION ENFORCEMENT"
# =============================================================================

info "Authorized: supervisor-agent → airline-agent-sidecar"
STATUS=$(curl -s -o /tmp/zta_resp.json -w "%{http_code}" -H "x-agent-id: supervisor-agent" ${AIRLINE_SIDECAR}/.well-known/agent.json)
echo "  Status: ${STATUS}"
cat /tmp/zta_resp.json | python3 -m json.tool 2>/dev/null || cat /tmp/zta_resp.json
[[ "$STATUS" == "200" ]] && pass "Authorized agent allowed" || fail "Should have been allowed"
divider

info "Unauthorized: evil-agent → airline-agent-sidecar"
STATUS=$(curl -s -o /tmp/zta_resp.json -w "%{http_code}" -H "x-agent-id: evil-agent" ${AIRLINE_SIDECAR}/.well-known/agent.json)
echo "  Status: ${STATUS}"
cat /tmp/zta_resp.json | python3 -m json.tool 2>/dev/null || cat /tmp/zta_resp.json
[[ "$STATUS" == "403" ]] && pass "Unauthorized agent blocked" || fail "Should have been 403"
divider

info "Missing identity → airline-agent-sidecar"
STATUS=$(curl -s -o /tmp/zta_resp.json -w "%{http_code}" ${AIRLINE_SIDECAR}/.well-known/agent.json)
echo "  Status: ${STATUS}"
cat /tmp/zta_resp.json | python3 -m json.tool 2>/dev/null || cat /tmp/zta_resp.json
[[ "$STATUS" == "403" ]] && pass "Missing identity blocked" || fail "Should have been 403"
divider

info "Cross-domain: airline-agent → hotel-mcp-sidecar (should be blocked)"
STATUS=$(curl -s -o /tmp/zta_resp.json -w "%{http_code}" -H "x-agent-id: airline-agent" ${HOTEL_SIDECAR}/.well-known/agent.json)
echo "  Status: ${STATUS}"
cat /tmp/zta_resp.json | python3 -m json.tool 2>/dev/null || cat /tmp/zta_resp.json
[[ "$STATUS" == "403" ]] && pass "Cross-domain blocked" || fail "Should have been 403"

# =============================================================================
header "4. OPA POLICY EVALUATION"
# =============================================================================

info "OPA: supervisor-agent → airline-agent (should allow)"
curl -s -X POST ${OPA_URL}/v1/data/zta/authz \
  -H "Content-Type: application/json" \
  -d '{"input":{"source":"supervisor-agent","destination":"airline-agent","path":"/.well-known/agent.json"}}' | python3 -m json.tool
divider

info "OPA: evil-agent → airline-agent (should deny)"
curl -s -X POST ${OPA_URL}/v1/data/zta/authz \
  -H "Content-Type: application/json" \
  -d '{"input":{"source":"evil-agent","destination":"airline-agent","path":"/.well-known/agent.json"}}' | python3 -m json.tool
divider

info "OPA: airline-agent → airline-mcp (should allow)"
curl -s -X POST ${OPA_URL}/v1/data/zta/authz \
  -H "Content-Type: application/json" \
  -d '{"input":{"source":"airline-agent","destination":"airline-mcp","path":"/sse"}}' | python3 -m json.tool
divider

info "OPA: airline-agent → hotel-mcp (should deny)"
curl -s -X POST ${OPA_URL}/v1/data/zta/authz \
  -H "Content-Type: application/json" \
  -d '{"input":{"source":"airline-agent","destination":"hotel-mcp","path":"/sse"}}' | python3 -m json.tool

# =============================================================================
header "5. JWT AUTHENTICATION"
# =============================================================================

info "Requesting JWT for supervisor-agent..."
TOKEN_RESP=$(curl -s -X POST ${AUTH_URL}/token \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "supervisor-agent", "secret": "zta-agent-shared-secret"}')
echo "$TOKEN_RESP" | python3 -m json.tool
TOKEN=$(echo "$TOKEN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")
divider

if [ -n "$TOKEN" ] && [ "$TOKEN" != "" ]; then
  info "Verifying JWT..."
  curl -s -X POST ${AUTH_URL}/token/verify \
    -H "Content-Type: application/json" \
    -d "{\"token\": \"$TOKEN\"}" | python3 -m json.tool
  pass "Token issued and verified"
else
  fail "Could not obtain token"
fi
divider

info "Requesting JWT with invalid secret..."
curl -s -X POST ${AUTH_URL}/token \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "supervisor-agent", "secret": "wrong-secret"}' | python3 -m json.tool

# =============================================================================
header "6. ZTA RESPONSE HEADERS"
# =============================================================================

info "Checking ZTA headers on sidecar response..."
curl -s -D - -H "x-agent-id: supervisor-agent" ${AIRLINE_SIDECAR}/health -o /dev/null 2>&1 | grep -i "x-zta\|x-zt"
echo ""

# =============================================================================
header "7. SIDECAR OVERHEAD BENCHMARK"
# =============================================================================

info "Direct agent call (no sidecar) — 10 requests..."
for i in $(seq 1 10); do
  curl -s -o /dev/null -w "%{time_total}" ${AIRLINE_DIRECT}/health
  echo ""
done | awk '{sum+=$1; count++} END {printf "  Avg: %.1fms\n", (sum/count)*1000}'

info "Via sidecar — 10 requests..."
for i in $(seq 1 10); do
  curl -s -o /dev/null -w "%{time_total}" -H "x-agent-id: supervisor-agent" ${AIRLINE_SIDECAR}/health
  echo ""
done | awk '{sum+=$1; count++} END {printf "  Avg: %.1fms\n", (sum/count)*1000}'

# =============================================================================
if [ "$QUICK" = false ]; then
header "8. LOAD TEST (via test driver)"
info "Running load test: 10 concurrent, 50 requests..."
cd test-driver 2>/dev/null && python3 main.py --mode load --concurrency 10 --requests 50 2>&1 || echo "  (test driver not found — run from repo root)"
cd - > /dev/null 2>/dev/null
fi

# =============================================================================
header "9. AUTOSCALER (dry run)"
# =============================================================================

info "Running autoscaler in dry-run mode..."
python3 services/autoscaler/autoscaler.py --once --dry-run 2>&1 || echo "  (autoscaler not found — run from repo root)"

# =============================================================================
header "DEMO COMPLETE"
# =============================================================================

echo -e "  ${GREEN}${BOLD}All demonstrations completed.${NC}"
echo -e "  ${DIM}System: 18 containers | 6 ZTA sidecars | OPA + JWT Auth${NC}"
echo -e "  ${DIM}Protocols: A2A (agent-to-agent) + MCP (agent-to-tool)${NC}"
echo ""