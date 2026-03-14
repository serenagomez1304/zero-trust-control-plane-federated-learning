# Zero Trust Control Plane for Multi-Agent Systems

A federated Zero-Trust Architecture (ZTA) control plane for a multi-agent AI system, implemented as an independent study. The project demonstrates how ZTA principles — never trust, always verify — can govern both Agent-to-Tool (MCP) and Agent-to-Agent (A2A) interactions across a distributed multi-agent travel planning system.

## Architecture

```
User
 └─▶ Supervisor (:8080)               LangGraph StateGraph · Groq LLM · JWT Auth
      │
      ├─▶ Airline Agent Sidecar (:19091)    .NET YARP · WAF · MicroSeg · OPA · DLP
      │    └─▶ Airline Agent (:8091)         A2A Server · LangGraph ReAct · MCP Client
      │         └─▶ Airline MCP Sidecar (:19010)
      │              └─▶ Airline MCP (:8010)  FastMCP SSE → Airline Service (:8001)
      │
      ├─▶ Hotel Agent Sidecar (:19092)
      │    └─▶ Hotel Agent (:8092)
      │         └─▶ Hotel MCP Sidecar (:19011)
      │              └─▶ Hotel MCP (:8011)    FastMCP SSE → Hotel Service (:8002)
      │
      └─▶ Car Rental Agent Sidecar (:19093)
           └─▶ Car Rental Agent (:8093)
                └─▶ Car Rental MCP Sidecar (:19012)
                     └─▶ Car Rental MCP (:8012)  FastMCP SSE → Car Rental Service (:8003)

 OPA (:8181)          Policy Decision Point — Rego policies
 ZTA Auth (:8180)     JWT Token Issuer — agent identity tokens
```

Every arrow passes through a ZTA sidecar that enforces WAF, micro-segmentation, OPA policy evaluation, DLP inspection, and JWT validation — true zero trust at every hop.

## What's Implemented

### Phase 1 — Base App (A2A + MCP + LangGraph)

- **A2A Protocol Library** (`agents/a2a/`) — Full implementation of Google's Agent2Agent Protocol: Agent Cards for capability discovery, Task lifecycle (submitted → working → completed/failed), Message/Part types, JSON-RPC 2.0 transport. 23 unit tests.
- **MCP Servers** — FastMCP with SSE transport, wrapping airline, hotel, and car rental backend services as tools for LLM agents.
- **Domain Agents** — Three specialized agents (airline, hotel, car-rental), each running a LangGraph ReAct tool-calling loop connected to its MCP server via `langchain-mcp-adapters`. Groq `llama-4-scout` as default LLM.
- **Supervisor** — LangGraph StateGraph that discovers agents via A2A Agent Cards at startup, classifies user intent with an LLM, and delegates tasks via A2A `tasks/send`.

### Phase 2 — ZTA Control Plane

- **.NET YARP Sidecars** — Single Docker image configured per-component via env vars. Replaces Envoy with a .NET reverse proxy running the full ZeroTrustModel middleware pipeline:
  - Security Monitoring (structured audit logging)
  - WAF (SQL injection, XSS, rate limiting)
  - JWT Authentication (Bearer token validation)
  - Micro-Segmentation (agent-to-agent access control via `ALLOWED_SOURCES`)
  - Policy Engine (calls OPA for authorization decisions)
  - DLP (data loss prevention headers)
- **OPA Policies** — Rego rules defining which agents can communicate with which targets. Supervisor can reach domain agents; domain agents can reach their MCP server; everything else is denied.
- **JWT Auth Service** — Issues signed JWT tokens containing agent identity, type, allowed targets, and roles. Agents authenticate at startup; tokens are validated by sidecars on every request.
- **Full Sidecar Mesh** — All traffic routed through sidecars: supervisor → agent sidecars → agents → MCP sidecars → MCP servers.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
git clone <repo-url>
cd zero-trust-control-plane

# Set your LLM key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# Start everything (18 containers)
docker compose -f docker-compose.zta.yml up --build
```

### Phase 1 Only (no sidecars, 10 containers)

```bash
docker compose -f docker-compose.zta.yml up --build \
  airline-service hotel-service car-rental-service \
  airline-mcp hotel-mcp car-rental-mcp \
  airline-agent hotel-agent car-rental-agent \
  supervisor
```

## Testing

### End-to-End Queries

```bash
# Flight search
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find flights from JFK to LAX on 2025-07-15"}' | python3 -m json.tool

# Hotel search
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Search for hotels in New York from July 15 to July 18"}' | python3 -m json.tool

# Car rental search
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find rental cars at LAX from July 15 to July 20"}' | python3 -m json.tool
```

### A2A Protocol

```bash
# Discover agent capabilities
curl -s http://localhost:8091/.well-known/agent.json | python3 -m json.tool

# Send a task directly via A2A JSON-RPC
curl -s -X POST http://localhost:8091/a2a \
  -H "Content-Type: application/json" \
  -H "x-agent-id: supervisor-agent" \
  -d '{
    "jsonrpc": "2.0", "id": "test-1", "method": "tasks/send",
    "params": {"id": "task-001", "message": {"role": "user", "parts": [{"type": "text", "text": "List all airports"}]}}
  }' | python3 -m json.tool
```

### ZTA Enforcement

```bash
# Sidecar health
curl -s http://localhost:19091/sidecar/health | python3 -m json.tool

# Allowed: supervisor accessing agent through sidecar
curl -s -H "x-agent-id: supervisor-agent" http://localhost:19091/.well-known/agent.json | python3 -m json.tool

# Blocked: unauthorized agent (returns 403)
curl -s -H "x-agent-id: evil-agent" http://localhost:19091/a2a | python3 -m json.tool

# Blocked: no identity header (returns 403)
curl -s http://localhost:19091/a2a | python3 -m json.tool

# ZTA headers on responses
curl -sI -H "x-agent-id: supervisor-agent" http://localhost:19091/health 2>&1 | grep "X-ZTA"
```

### JWT Authentication

```bash
# Issue a token
curl -s -X POST http://localhost:8180/token \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "supervisor-agent", "secret": "zta-agent-shared-secret"}' | python3 -m json.tool

# Verify a token
TOKEN=$(curl -s -X POST http://localhost:8180/token -H "Content-Type: application/json" \
  -d '{"agent_id": "supervisor-agent", "secret": "zta-agent-shared-secret"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -s -X POST http://localhost:8180/token/verify \
  -H "Content-Type: application/json" \
  -d "{\"token\": \"$TOKEN\"}" | python3 -m json.tool
```

### OPA Policy Decisions

```bash
# Allowed: supervisor → airline-agent
curl -s -X POST http://localhost:8181/v1/data/zta/authz/allow \
  -H "Content-Type: application/json" \
  -d '{"input": {"agent_id": "supervisor-agent", "path": "/a2a", "method": "POST", "component": "airline-agent-sidecar"}}' | python3 -m json.tool

# Denied: evil-agent → airline-agent
curl -s -X POST http://localhost:8181/v1/data/zta/authz/allow \
  -H "Content-Type: application/json" \
  -d '{"input": {"agent_id": "evil-agent", "path": "/a2a", "method": "POST", "component": "airline-agent-sidecar"}}' | python3 -m json.tool
```

### Benchmarks

```bash
pip install httpx
python benchmarks/latency_benchmark.py --runs 10 --suite all
```

### Unit Tests

```bash
pip install -e ".[test]"
pytest tests/test_a2a.py -v
```

## Project Structure

```
zero-trust-control-plane/
├── agents/
│   ├── __init__.py
│   ├── a2a/                        # A2A Protocol Library
│   │   ├── models.py               #   Pydantic models (AgentCard, Task, Message, etc.)
│   │   ├── server.py               #   A2AServer FastAPI router
│   │   └── client.py               #   A2AClient async HTTP client + JWT auth
│   ├── airline-agent/              # Domain agents (A2A + LangGraph + MCP)
│   ├── hotel-agent/
│   ├── car-rental-agent/
│   └── supervisor/                 # LangGraph StateGraph orchestrator
├── mcp-servers/
│   ├── airline/                    # FastMCP SSE servers
│   ├── hotel/
│   └── car-rental/
├── services/
│   ├── airline/                    # Backend services (FastAPI + SQLite)
│   ├── hotel/
│   ├── car-rental/
│   └── auth/                       # JWT Token Issuer
├── zta-sidecar/                    # .NET YARP reverse proxy
│   ├── Program.cs                  #   Full ZTA middleware pipeline
│   ├── ZtaSidecar.csproj
│   └── Dockerfile
├── zta-infrastructure/
│   └── opa/
│       └── policy.rego             # Agent-to-agent access control policies
├── benchmarks/
│   └── latency_benchmark.py        # Latency with/without ZTA measurement
├── tests/
│   └── test_a2a.py                 # 23 unit tests for A2A library
├── docker-compose.zta.yml          # Full deployment (18 containers)
└── pyproject.toml
```

## Port Reference

| Component | Direct Port | Sidecar Port |
|-----------|------------|--------------|
| Supervisor | 8080 | — |
| Airline Agent | 8091 | 19091 |
| Hotel Agent | 8092 | 19092 |
| Car Rental Agent | 8093 | 19093 |
| Airline MCP | 8010 | 19010 |
| Hotel MCP | 8011 | 19011 |
| Car Rental MCP | 8012 | 19012 |
| OPA | 8181 | — |
| Auth Service | 8180 | — |
| Backend Services | 8001-8003 | — |

## Benchmark Results

ZTA sidecar overhead measured via `benchmarks/latency_benchmark.py`:

| Path | Mean Latency | P95 |
|------|-------------|-----|
| A2A Direct (agent :8091) | ~13ms | ~25ms |
| A2A via Sidecar (:19091) | ~16ms | ~32ms |
| **ZTA Overhead** | **~3ms** | **~7ms** |

End-to-end (user → supervisor → sidecar → agent → MCP → backend):

| Query | Mean | P95 |
|-------|------|-----|
| Flight Search | ~185ms | ~319ms |
| Hotel Search | ~157ms | ~331ms |
| Car Rental Search | ~162ms | ~428ms |

## License

MIT License