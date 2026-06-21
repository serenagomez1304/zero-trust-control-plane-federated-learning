"""
Tests for the A2A Protocol Library
====================================
Validates:
- Model serialization/deserialization
- Agent Card structure
- Task lifecycle state transitions
- JSON-RPC 2.0 envelope handling
- A2AServer JSON-RPC dispatch (using FastAPI TestClient)
- A2AClient ↔ A2AServer round-trip
"""

import asyncio
import json
import sys
import os
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.a2a.models import (
    A2AError,
    AgentAuthentication,
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    Artifact,
    DataPart,
    FilePart,
    FileContent,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    Task,
    TaskCancelParams,
    TaskQueryParams,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
)
from agents.a2a.server import A2AServer
from agents.a2a.trust import (
    Attestation,
    Content,
    ContextEntry,
    DeclaredPurpose,
    Kappa,
    Mu,
    Sigma,
    StubSemanticModel,
    TrustMessage,
    resolve_mu,
)
from agents.a2a.substrate import (
    ChainAgent,
    TRUST_THRESHOLD,
    originate,
    process_hop,
    run_chain,
    synthesize,
    verify_signature,
)


# =============================================================================
# Fixtures
# =============================================================================

def make_agent_card() -> AgentCard:
    """Create a test Agent Card."""
    return AgentCard(
        name="Test Airline Agent",
        description="Handles flight search and booking",
        url="http://localhost:8091",
        version="1.0.0",
        provider=AgentProvider(organization="ZTA Testbed"),
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=True,
        ),
        authentication=AgentAuthentication(schemes=["bearer"]),
        skills=[
            AgentSkill(
                id="flight-search",
                name="Flight Search",
                description="Search for flights between airports",
                tags=["flights", "airline", "travel"],
                examples=["Find flights from JFK to LAX on 2025-06-15"],
            ),
            AgentSkill(
                id="flight-booking",
                name="Flight Booking",
                description="Book a flight for passengers",
                tags=["booking", "reservation"],
                examples=["Book flight AA123 for John Doe"],
            ),
        ],
    )


async def echo_handler(task: Task, message: Message) -> Task:
    """Simple handler that echoes the input as an artifact."""
    # Extract text from the message
    text_parts = [p.text for p in message.parts if isinstance(p, TextPart)]
    response_text = f"Processed: {' '.join(text_parts)}"

    # Set completed status with response
    task.status = TaskStatus(
        state=TaskState.COMPLETED,
        message=Message(
            role="agent",
            parts=[TextPart(text=response_text)],
        ),
    )

    # Add artifact
    if task.artifacts is None:
        task.artifacts = []
    task.artifacts.append(
        Artifact(
            name="response",
            parts=[TextPart(text=response_text)],
            index=len(task.artifacts),
        )
    )

    return task


def make_test_app() -> FastAPI:
    """Create a FastAPI app with the A2A server for testing."""
    card = make_agent_card()
    server = A2AServer(card=card, handler=echo_handler)
    app = FastAPI()
    app.include_router(server.router)
    return app


# =============================================================================
# Model Tests
# =============================================================================

class TestModels:
    """Test Pydantic model serialization and validation."""

    def test_text_part(self):
        p = TextPart(text="Hello")
        d = p.model_dump()
        assert d == {"type": "text", "text": "Hello"}

    def test_data_part(self):
        p = DataPart(data={"flights": [{"id": "AA123"}]})
        d = p.model_dump()
        assert d["type"] == "data"
        assert d["data"]["flights"][0]["id"] == "AA123"

    def test_file_part(self):
        p = FilePart(file=FileContent(name="ticket.pdf", mimeType="application/pdf", uri="https://example.com/ticket.pdf"))
        d = p.model_dump()
        assert d["type"] == "file"
        assert d["file"]["name"] == "ticket.pdf"

    def test_message(self):
        m = Message(
            role="user",
            parts=[TextPart(text="Search flights JFK to LAX")],
            metadata={"conversation_id": "abc123"},
        )
        d = m.model_dump()
        assert d["role"] == "user"
        assert len(d["parts"]) == 1
        assert d["parts"][0]["text"] == "Search flights JFK to LAX"

    def test_task_state_terminal(self):
        assert TaskState.COMPLETED.is_terminal is True
        assert TaskState.FAILED.is_terminal is True
        assert TaskState.CANCELED.is_terminal is True
        assert TaskState.WORKING.is_terminal is False
        assert TaskState.SUBMITTED.is_terminal is False
        assert TaskState.INPUT_REQUIRED.is_terminal is False

    def test_task_default(self):
        t = Task()
        assert t.status.state == TaskState.SUBMITTED
        assert t.id is not None
        assert len(t.id) == 36  # UUID format

    def test_task_send_params(self):
        p = TaskSendParams(
            id="task-1",
            message=Message(role="user", parts=[TextPart(text="hello")]),
            metadata={"x-agent-id": "supervisor"},
        )
        d = p.model_dump()
        assert d["id"] == "task-1"
        assert d["message"]["role"] == "user"

    def test_agent_card_serialization(self):
        card = make_agent_card()
        d = card.model_dump(exclude_none=True)
        
        assert d["name"] == "Test Airline Agent"
        assert d["protocolVersion"] == "0.2"
        assert len(d["skills"]) == 2
        assert d["skills"][0]["id"] == "flight-search"
        assert d["authentication"]["schemes"] == ["bearer"]
        assert d["capabilities"]["stateTransitionHistory"] is True

    def test_agent_card_roundtrip(self):
        """Card serializes to JSON and deserializes back identically."""
        card = make_agent_card()
        json_str = card.model_dump_json()
        card2 = AgentCard.model_validate_json(json_str)
        assert card2.name == card.name
        assert len(card2.skills) == len(card.skills)
        assert card2.skills[0].id == card.skills[0].id

    def test_jsonrpc_request(self):
        req = JSONRPCRequest(
            method="tasks/send",
            params={"id": "t1", "message": {"role": "user", "parts": [{"type": "text", "text": "hi"}]}},
        )
        d = req.model_dump()
        assert d["jsonrpc"] == "2.0"
        assert d["method"] == "tasks/send"

    def test_jsonrpc_response_success(self):
        resp = JSONRPCResponse(id="1", result={"status": "ok"})
        d = resp.model_dump(exclude_none=True)
        assert "error" not in d
        assert d["result"]["status"] == "ok"

    def test_jsonrpc_response_error(self):
        resp = JSONRPCResponse(
            id="1",
            error=JSONRPCError(code=-32001, message="Task not found"),
        )
        d = resp.model_dump(exclude_none=True)
        assert "result" not in d
        assert d["error"]["code"] == -32001

    def test_a2a_error_helpers(self):
        err = A2AError.task_not_found("task-xyz")
        assert err.code == -32001
        assert "task-xyz" in err.message

        err2 = A2AError.policy_violation("rate_limit_exceeded")
        assert err2.code == -32053


# =============================================================================
# Server Tests (via FastAPI TestClient)
# =============================================================================

class TestA2AServer:
    """Test A2A server endpoints using sync TestClient."""

    def setup_method(self):
        self.app = make_test_app()
        self.client = TestClient(self.app)

    def test_agent_card_endpoint(self):
        """GET /.well-known/agent.json returns the Agent Card."""
        resp = self.client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test Airline Agent"
        assert len(data["skills"]) == 2
        assert data["protocolVersion"] == "0.2"

    def test_a2a_health(self):
        """GET /a2a/health returns status."""
        resp = self.client.get("/a2a/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["protocol"] == "a2a"

    def test_tasks_send_new(self):
        """POST /a2a with tasks/send creates and completes a task."""
        task_id = str(uuid.uuid4())
        resp = self.client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "rpc-1",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Search flights JFK to LAX"}],
                },
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == "rpc-1"
        assert "error" not in body

        result = body["result"]
        assert result["id"] == task_id
        assert result["status"]["state"] == "completed"
        assert len(result["artifacts"]) == 1
        assert "Processed: Search flights JFK to LAX" in result["artifacts"][0]["parts"][0]["text"]

    def test_tasks_get(self):
        """Submit a task, then query it via tasks/get."""
        task_id = str(uuid.uuid4())

        # Submit
        self.client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "rpc-1",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "hello"}],
                },
            },
        })

        # Query
        resp = self.client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "rpc-2",
            "method": "tasks/get",
            "params": {"id": task_id},
        })
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["id"] == task_id
        assert result["status"]["state"] == "completed"

    def test_tasks_get_not_found(self):
        """tasks/get for non-existent task returns error."""
        resp = self.client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "rpc-3",
            "method": "tasks/get",
            "params": {"id": "nonexistent"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["error"]["code"] == -32001

    def test_tasks_cancel(self):
        """Cancel a completed task should return TASK_NOT_CANCELABLE."""
        task_id = str(uuid.uuid4())

        # Submit (handler completes immediately)
        self.client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "rpc-1",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {"role": "user", "parts": [{"type": "text", "text": "test"}]},
            },
        })

        # Try to cancel
        resp = self.client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "rpc-4",
            "method": "tasks/cancel",
            "params": {"id": task_id},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["error"]["code"] == -32002  # TASK_NOT_CANCELABLE

    def test_method_not_found(self):
        """Unknown method returns METHOD_NOT_FOUND."""
        resp = self.client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "rpc-5",
            "method": "tasks/unknown",
            "params": {},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["error"]["code"] == -32601

    def test_invalid_json(self):
        """Malformed body returns PARSE_ERROR."""
        resp = self.client.post(
            "/a2a",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["error"]["code"] == -32700

    def test_zta_headers_propagated(self):
        """ZTA headers are captured in task metadata."""
        task_id = str(uuid.uuid4())
        resp = self.client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "rpc-6",
                "method": "tasks/send",
                "params": {
                    "id": task_id,
                    "message": {"role": "user", "parts": [{"type": "text", "text": "test"}]},
                },
            },
            headers={
                "x-agent-id": "supervisor-agent",
                "x-request-id": "req-12345",
                "x-trust-score": "0.95",
            },
        )
        result = resp.json()["result"]
        meta = result.get("metadata", {})
        assert meta.get("x-agent-id") == "supervisor-agent"
        assert meta.get("x-trust-score") == "0.95"

    def test_task_history_recorded(self):
        """Verify state transition history is recorded."""
        task_id = str(uuid.uuid4())
        resp = self.client.post("/a2a", json={
            "jsonrpc": "2.0",
            "id": "rpc-7",
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "message": {"role": "user", "parts": [{"type": "text", "text": "test"}]},
            },
        })
        result = resp.json()["result"]
        history = result.get("history", [])
        # Should have: submitted, working, completed (from handler), plus the post-handler append
        states = [h["state"] for h in history]
        assert "submitted" in states
        assert "working" in states
        assert "completed" in states


# =============================================================================
# Per-Message Trust Model — Message Format (trust.py)
# =============================================================================

def make_chain_agents():
    """A stub user -> supervisor -> airline-agent chain."""
    supervisor = ChainAgent(
        agent_id="supervisor",
        declared_purpose=DeclaredPurpose(label="route-travel", description="route to a specialist"),
        attestation=Attestation(agent_id="supervisor"),
        handler=lambda view: "Routing JFK->LAX request to the airline specialist",
    )
    airline = ChainAgent(
        agent_id="airline-agent",
        declared_purpose=DeclaredPurpose(label="flight-search", description="search and book flights"),
        attestation=Attestation(agent_id="airline-agent"),
        handler=lambda view: "Booked flight AA123 JFK->LAX on 2025-07-01, confirmation XJ7F2",
    )
    return supervisor, airline


class TestTrustMessageModels:
    """The message format m = <content, mu, kappa, sigma>."""

    def test_trust_message_construction(self):
        m = TrustMessage(
            content=Content(payload="hello"),
            mu=Mu(),
            kappa=Kappa(intent="hello"),
            sigma=Sigma(chain=["stub-sig:abc"]),
        )
        assert m.content.payload == "hello"
        assert m.kappa.intent == "hello"
        assert m.mu.bundle == ["trust_eval", "transform", "integrate", "synthesize"]

    def test_trust_message_roundtrip(self):
        m = originate("Book a flight JFK to LAX")
        m2 = TrustMessage.model_validate_json(m.model_dump_json())
        assert m2.content.payload == m.content.payload
        assert m2.kappa.intent == m.kappa.intent
        assert m2.sigma.chain == m.sigma.chain

    def test_mu_no_protected_namespace_warning(self):
        """Mu.model_id must not trip Pydantic's protected 'model_' namespace."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mu = Mu(model_id="custom")
        assert mu.model_id == "custom"

    def test_kappa_starts_empty(self):
        m = originate("intent text")
        assert m.kappa.intent == "intent text"
        assert m.kappa.entries == []


class TestStubSemanticModel:
    """The four stubbed capabilities of mu (Milestone 1)."""

    def setup_method(self):
        self.mu = StubSemanticModel()
        self.purpose = DeclaredPurpose(label="flight-search")
        self.att = Attestation(agent_id="airline-agent")
        self.kappa = Kappa(intent="book a flight")

    def test_trust_eval_always_one(self):
        assert self.mu.trust_eval(self.purpose, self.att, self.kappa) == 1.0

    def test_transform_returns_content_unchanged(self):
        content = Content(payload="raw payload", data={"k": "v"})
        view = self.mu.transform(content, self.purpose, self.kappa)
        assert view == content

    def test_integrate_appends_response(self):
        k2 = self.mu.integrate("a response", self.kappa, self.purpose)
        assert len(k2.entries) == 1
        assert k2.entries[0] == ContextEntry(purpose="flight-search", response="a response")
        # original kappa is not mutated
        assert self.kappa.entries == []

    def test_synthesize_returns_latest_response(self):
        k = Kappa(intent="x", entries=[
            ContextEntry(purpose="route-travel", response="first"),
            ContextEntry(purpose="flight-search", response="second"),
        ])
        assert self.mu.synthesize(k) == "second"

    def test_synthesize_empty_context(self):
        assert self.mu.synthesize(Kappa(intent="x")) == ""

    def test_resolve_mu_returns_stub(self):
        assert isinstance(resolve_mu(Mu()), StubSemanticModel)


# =============================================================================
# Per-Message Trust Model — Substrate Pipeline (substrate.py)
# =============================================================================

class TestSubstratePipeline:
    """The per-hop pipeline: verify -> trust_eval -> transform -> integrate."""

    def test_originate_signs_initial_bundle(self):
        m = originate("an intent")
        assert len(m.sigma.chain) == 1
        assert verify_signature(m) is True

    def test_verify_detects_tampering(self):
        m = originate("an intent")
        m.content.payload = "tampered"
        assert verify_signature(m) is False

    def test_verify_empty_chain_fails(self):
        m = originate("an intent")
        m.sigma.chain = []
        assert verify_signature(m) is False

    def test_process_hop_accepts_and_resigns(self):
        supervisor, _ = make_chain_agents()
        m = originate("Book a flight")
        result = process_hop(m, supervisor)

        assert result.accepted is True
        assert result.trust == 1.0
        # agent saw the (unchanged) view
        assert result.view.payload == "Book a flight"
        assert result.response.startswith("Routing")
        # resigned message: new context entry + extended, valid signature chain
        assert len(result.message.kappa.entries) == 1
        assert len(result.message.sigma.chain) == 2
        assert verify_signature(result.message) is True

    def test_process_hop_rejects_below_threshold(self):
        supervisor, _ = make_chain_agents()
        m = originate("Book a flight")
        # Stub trust_eval returns 1.0; a threshold above it forces the reject path.
        result = process_hop(m, supervisor, threshold=1.5)
        assert result.accepted is False
        assert result.message is None
        assert "below threshold" in result.reason

    def test_process_hop_rejects_tampered_message(self):
        supervisor, _ = make_chain_agents()
        m = originate("Book a flight")
        m.content.payload = "tampered after signing"
        result = process_hop(m, supervisor)
        assert result.accepted is False
        assert "signature" in result.reason


class TestEndToEndChain:
    """user -> supervisor -> airline-agent end-to-end (Milestone 1 acceptance)."""

    def test_full_chain_completes_and_synthesizes(self):
        supervisor, airline = make_chain_agents()
        m = originate("Book a flight from JFK to LAX on 2025-07-01")
        result = run_chain(m, [supervisor, airline])

        assert result.completed is True
        assert result.rejected_at is None
        # one accepted hop per agent
        assert [h.agent_id for h in result.hops] == ["supervisor", "airline-agent"]
        assert all(h.accepted for h in result.hops)
        # context accumulated one entry per hop, in order
        purposes = [e.purpose for e in result.final_message.kappa.entries]
        assert purposes == ["route-travel", "flight-search"]
        # signature chain: origin + one per hop, still valid
        assert len(result.final_message.sigma.chain) == 3
        assert verify_signature(result.final_message) is True
        # final response synthesized from accumulated context (latest response, per stub)
        assert result.response == "Booked flight AA123 JFK->LAX on 2025-07-01, confirmation XJ7F2"
        assert result.response == synthesize(result.final_message)

    def test_chain_stops_at_rejected_hop(self):
        supervisor, airline = make_chain_agents()
        m = originate("Book a flight")
        # Threshold above the stub's 1.0 rejects the very first hop.
        result = run_chain(m, [supervisor, airline], threshold=1.5)
        assert result.completed is False
        assert result.rejected_at == "supervisor"
        assert len(result.hops) == 1
        assert result.response is None

    def test_intent_preserved_through_chain(self):
        supervisor, airline = make_chain_agents()
        m = originate("Book a flight from JFK to LAX")
        result = run_chain(m, [supervisor, airline])
        assert result.final_message.kappa.intent == "Book a flight from JFK to LAX"
        # raw content is never mutated by the chain
        assert result.final_message.content.payload == "Book a flight from JFK to LAX"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    # Run with pytest if available, else manual
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
