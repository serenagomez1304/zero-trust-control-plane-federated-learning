#!/usr/bin/env bash
# =============================================================================
# fix_cors.sh — patch the four federated control-plane services with CORS
#
# Why: when the demo UI is opened from file:// or any non-localhost-:8080
# origin, the browser issues a CORS preflight (OPTIONS) before POST/GET. The
# four newer services (trust-scorer, audit-logger, revocation-dispatcher,
# pdp-behavior) don't have CORS middleware, so the preflight returns 405 and
# the UI shows "Load failed".
#
# What this script does:
#   1. Add `from fastapi.middleware.cors import CORSMiddleware` to each server
#   2. Inject `app.add_middleware(CORSMiddleware, ...)` after `app = FastAPI(...)`
#   3. Rebuild only the four affected containers
#   4. Wait for healthchecks
#   5. Verify the OPTIONS preflight now returns 200
#
# Idempotent: re-running it after a successful patch is a no-op.
#
# Usage:
#   chmod +x fix_cors.sh
#   ./fix_cors.sh
#
# Run from the project root (the directory containing docker-compose.zta.yml).
# =============================================================================

set -euo pipefail

# ── Sanity checks ────────────────────────────────────────────────────────────

if [[ ! -f docker-compose.zta.yml ]]; then
  echo "ERROR: docker-compose.zta.yml not found in $(pwd)"
  echo "Run this script from the zero-trust-control-plane project root."
  exit 1
fi

SERVICES=(
  "services/trust-scorer/server.py"
  "services/audit-logger/server.py"
  "services/revocation-dispatcher/server.py"
  "services/pdp-behavior/server.py"
)

for f in "${SERVICES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: expected file not found: $f"
    exit 1
  fi
done

echo "─── Step 1/5 · Patching server.py files ──────────────────────────────"

for f in "${SERVICES[@]}"; do
  python3 - <<PY
import re, sys
p = "$f"
s = open(p).read()
changed = False

# Skip files that already have CORS configured
if "CORSMiddleware" in s and "add_middleware" in s:
    print(f"  SKIP {p} (already patched)")
    sys.exit(0)

# 1. Add the CORS import after the existing 'from fastapi import ...' line
import_pat = re.compile(r"^(from fastapi import [^\n]+)$", re.M)
m = import_pat.search(s)
if not m:
    print(f"  WARN {p}: couldn't locate 'from fastapi import ...' line")
    sys.exit(1)
if "from fastapi.middleware.cors import CORSMiddleware" not in s:
    s = s[:m.end()] + "\nfrom fastapi.middleware.cors import CORSMiddleware" + s[m.end():]
    changed = True

# 2. Inject the middleware after the 'app = FastAPI(...)' block (which can span
#    multiple lines). Match from 'app = FastAPI(' through the matching ')'.
app_pat = re.compile(r"(app\s*=\s*FastAPI\([\s\S]*?\n\))", re.M)
m = app_pat.search(s)
if not m:
    print(f"  WARN {p}: couldn't locate 'app = FastAPI(...)' block")
    sys.exit(1)
insertion = '''

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)'''
if "add_middleware(\n    CORSMiddleware" not in s:
    s = s[:m.end()] + insertion + s[m.end():]
    changed = True

if changed:
    open(p, "w").write(s)
    print(f"  PATCHED {p}")
else:
    print(f"  SKIP {p} (already had middleware)")
PY
done

echo
echo "─── Step 2/5 · Rebuilding the four services ──────────────────────────"
docker compose -f docker-compose.zta.yml up -d --build \
  trust-scorer audit-logger revocation-dispatcher pdp-behavior

echo
echo "─── Step 3/5 · Waiting for healthchecks (up to 60 seconds) ────────────"
deadline=$((SECONDS + 60))
ok=0
while [[ $SECONDS -lt $deadline ]]; do
  ok=0
  for port in 8190 8195 8196 8197; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      ok=$((ok+1))
    fi
  done
  if [[ $ok -eq 4 ]]; then
    echo "  All 4 services healthy."
    break
  fi
  sleep 2
done

if [[ $ok -ne 4 ]]; then
  echo "  WARN: not all services healthy after 60s. Continuing anyway —"
  echo "        the verification step will tell us if CORS is fixed."
fi

echo
echo "─── Step 4/5 · Verifying CORS preflight (OPTIONS) ─────────────────────"
declare -A CHECK=(
  ["trust-scorer:8190"]="POST /score"
  ["audit-logger:8195"]="GET /events"
  ["revocation-dispatcher:8196"]="POST /quarantine/agent"
  ["pdp-behavior:8197"]="GET /agents/foo/decision"
)

all_ok=1
for key in "${!CHECK[@]}"; do
  name="${key%:*}"
  port="${key#*:}"
  method_path="${CHECK[$key]}"
  method="${method_path% *}"
  path="${method_path#* }"
  code=$(curl -s -o /dev/null -w "%{http_code}" \
    -X OPTIONS "http://localhost:${port}${path}" \
    -H "Origin: http://localhost" \
    -H "Access-Control-Request-Method: ${method}" \
    -H "Access-Control-Request-Headers: content-type")
  if [[ "$code" == "200" || "$code" == "204" ]]; then
    printf "  ✓ %-22s OPTIONS %-25s → %s\n" "$name" "$path" "$code"
  else
    printf "  ✗ %-22s OPTIONS %-25s → %s\n" "$name" "$path" "$code"
    all_ok=0
  fi
done

echo
echo "─── Step 5/5 · Summary ────────────────────────────────────────────────"
if [[ $all_ok -eq 1 ]]; then
  cat <<'EOF'
  ✓ CORS preflight succeeds on all four services.

  Next steps:
    1. Hard-reload the demo UI in your browser:
         macOS: Cmd+Shift+R     Linux/Win: Ctrl+Shift+R
    2. Click through the Security Analyst persona to confirm the
       trust-scorer / audit / behavior-PDP / revocation panels work.
    3. Click through the Repeat Offender's federated loop end-to-end
       as a dry-run before you record.
EOF
else
  cat <<'EOF'
  ✗ Some preflights still failed. Check container logs:
       docker compose -f docker-compose.zta.yml logs trust-scorer audit-logger \
                                                    revocation-dispatcher pdp-behavior \
         | tail -60

  Common causes:
    - Container failed to start after rebuild (check 'docker compose ps')
    - Patch was applied but rebuild didn't pick up the new file
      → try: docker compose -f docker-compose.zta.yml build --no-cache <service>
EOF
  exit 1
fi
