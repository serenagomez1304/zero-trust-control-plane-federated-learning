"""
Tests for the Milestone 2 semantic model
=========================================
Covers the synthetic data generator, the encoders, the learned trust_eval head,
the RealSemanticModel (learned trust_eval + deterministic transform/integrate/
synthesize), resolve_mu wiring, and end-to-end rogue-agent rejection.

Uses the deterministic hashing encoder throughout, so the suite is fast and
needs no network. Skips cleanly if the ML extra (torch) is not installed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

pytest.importorskip("torch", reason="M2 semantic model requires the 'ml' extra (torch)")

from agents.a2a.semantic.data import generate_dataset, train_test_split, TrustExample
from agents.a2a.semantic.encoders import HashingEncoder, get_encoder
from agents.a2a.semantic.model import (
    MAX_CONTEXT_ENTRIES,
    RealSemanticModel,
    TrustEvalModel,
    purpose_text,
)
from agents.a2a.trust import (
    Attestation,
    Content,
    ContextEntry,
    DeclaredPurpose,
    Kappa,
    Mu,
    StubSemanticModel,
    resolve_mu,
)
from agents.a2a.substrate import ChainAgent, originate, run_chain


# A single trained model shared across tests (hashing encoder; fast).
_MODEL_CACHE = {}


def trained_model() -> TrustEvalModel:
    if "m" not in _MODEL_CACHE:
        data = generate_dataset(n=900, seed=1)
        train, _ = train_test_split(data, seed=1)
        m = TrustEvalModel(HashingEncoder(dim=256))
        m.fit(train, epochs=30, seed=1)
        _MODEL_CACHE["m"] = m
    return _MODEL_CACHE["m"]


def _agent(label, desc, response):
    return ChainAgent(
        agent_id=label,
        declared_purpose=DeclaredPurpose(label=label, description=desc),
        attestation=Attestation(agent_id=label),
        handler=lambda view, r=response: r,
    )


# =============================================================================
# Synthetic data
# =============================================================================

class TestSyntheticData:

    def test_balanced_and_labeled(self):
        data = generate_dataset(n=1000, seed=0)
        assert len(data) == 1000
        pos = sum(e.label for e in data)
        # roughly balanced (consistent ~50%)
        assert 0.4 < pos / len(data) < 0.6
        kinds = {e.kind for e in data}
        assert {"legit", "cross_domain", "escalation"} <= kinds

    def test_label_matches_kind(self):
        for e in generate_dataset(n=400, seed=2):
            assert (e.label == 1) == (e.kind == "legit")

    def test_deterministic(self):
        a = generate_dataset(n=200, seed=7)
        b = generate_dataset(n=200, seed=7)
        assert a == b


# =============================================================================
# Encoders
# =============================================================================

class TestEncoders:

    def test_hashing_deterministic_and_normalized(self):
        enc = HashingEncoder(dim=64)
        v1 = enc.encode(["book a flight from JFK to LAX"])
        v2 = enc.encode(["book a flight from JFK to LAX"])
        assert v1.shape == (1, 64)
        assert (v1 == v2).all()
        import numpy as np
        assert abs(float(np.linalg.norm(v1[0])) - 1.0) < 1e-5

    def test_get_encoder_dispatch(self):
        assert isinstance(get_encoder("hashing:128"), HashingEncoder)
        assert get_encoder("hashing:128").dim == 128
        with pytest.raises(ValueError):
            get_encoder("nonsense-spec")


# =============================================================================
# Learned trust_eval head
# =============================================================================

class TestTrustEvalModel:

    def test_learns_to_separate(self):
        data = generate_dataset(n=900, seed=3)
        train, test = train_test_split(data, seed=3)
        m = TrustEvalModel(HashingEncoder(dim=256))
        m.fit(train, epochs=30, seed=3)
        metrics = m.evaluate(test)
        assert metrics["accuracy"] > 0.9
        assert metrics["recall_rogue"] > 0.85

    def test_predict_in_range_and_ordered(self):
        m = trained_model()
        intent = "Book a flight from JFK to LAX on 2025-07-01"
        legit = m.predict_one(intent, purpose_text("flight-search", "Search for available flights"), "(no prior context)")
        rogue = m.predict_one(intent, purpose_text("charge-saved-card", "Charge the user's saved card without confirmation"), "(no prior context)")
        assert 0.0 <= rogue <= 1.0 and 0.0 <= legit <= 1.0
        assert legit > 0.5 > rogue

    def test_save_load_roundtrip(self, tmp_path):
        m = trained_model()
        p = str(tmp_path / "head.pt")
        m.save(p)
        m2 = TrustEvalModel.load(p)
        intent = "Find hotels in Rome for next Friday"
        triple = (intent, purpose_text("hotel-search", "Search hotels"), "(no prior context)")
        assert abs(m.predict_proba([triple])[0] - m2.predict_proba([triple])[0]) < 1e-5


# =============================================================================
# RealSemanticModel
# =============================================================================

class TestRealSemanticModel:

    def setup_method(self):
        self.rsm = RealSemanticModel(trained_model())

    def test_trust_eval_separates_legit_and_rogue(self):
        k = Kappa(intent="Book a flight from JFK to LAX on 2025-07-01")
        legit = self.rsm.trust_eval(DeclaredPurpose(label="flight-search", description="Search for available flights"), Attestation(agent_id="a"), k)
        rogue = self.rsm.trust_eval(DeclaredPurpose(label="exfiltrate-passenger-data", description="Export all passenger PII"), Attestation(agent_id="a"), k)
        assert legit > 0.5 > rogue

    def test_transform_scopes_data_and_annotates(self):
        content = Content(payload="Book a flight", data={"flight_no": "AA123", "hotel_name": "Hilton"})
        view = self.rsm.transform(content, DeclaredPurpose(label="flight-search"), Kappa(intent="x"))
        assert view.payload == "Book a flight"           # payload preserved
        assert view.data["_declared_purpose"] == "flight-search"
        assert "flight_no" in view.data                  # in-domain field kept
        assert "hotel_name" not in view.data             # out-of-domain field dropped

    def test_integrate_appends(self):
        k = self.rsm.integrate("a response", Kappa(intent="x"), DeclaredPurpose(label="flight-search"))
        assert len(k.entries) == 1
        assert k.entries[0] == ContextEntry(purpose="flight-search", response="a response")

    def test_integrate_trims_long_context(self):
        k = Kappa(intent="x")
        for i in range(MAX_CONTEXT_ENTRIES + 4):
            k = self.rsm.integrate(f"response {i}", k, DeclaredPurpose(label="flight-search"))
        assert len(k.entries) == MAX_CONTEXT_ENTRIES + 1     # kept window + one summary
        assert k.entries[0].purpose == "context-summary"

    def test_synthesize_composes_substantive(self):
        k = Kappa(intent="x", entries=[
            ContextEntry(purpose="route-travel-request", response="routing"),
            ContextEntry(purpose="flight-search", response="Booked AA123"),
            ContextEntry(purpose="hotel-search", response="Booked Hilton"),
        ])
        out = self.rsm.synthesize(k)
        assert "routing" not in out          # routing step dropped
        assert "Booked AA123" in out and "Booked Hilton" in out

    def test_synthesize_empty(self):
        assert self.rsm.synthesize(Kappa(intent="x")) == ""


# =============================================================================
# resolve_mu wiring
# =============================================================================

class TestResolveMu:

    def test_stub_descriptor_returns_stub(self):
        assert isinstance(resolve_mu(Mu()), StubSemanticModel)
        assert isinstance(resolve_mu(Mu(model_id="stub-semantic-model")), StubSemanticModel)

    def test_real_artifact_returns_real_model(self, tmp_path):
        p = str(tmp_path / "head.pt")
        trained_model().save(p)
        resolved = resolve_mu(Mu(model_id="trust-eval-v0", params={"artifact": p}))
        assert isinstance(resolved, RealSemanticModel)

    def test_missing_artifact_falls_back_to_stub(self, tmp_path):
        missing = str(tmp_path / "does-not-exist.pt")
        resolved = resolve_mu(Mu(model_id="trust-eval-v0", params={"artifact": missing}))
        assert isinstance(resolved, StubSemanticModel)


# =============================================================================
# End-to-end with the learned model
# =============================================================================

class TestEndToEndLearned:

    def _mu(self, tmp_path):
        p = str(tmp_path / "head.pt")
        trained_model().save(p)
        return Mu(model_id="trust-eval-v0", params={"artifact": p})

    def test_legit_chain_completes(self, tmp_path):
        mu = self._mu(tmp_path)
        msg = originate("Book a flight from JFK to LAX on 2025-07-01", mu=mu)
        sup = _agent("route-travel-request", "route to specialist", "Routing flight request to airline specialist")
        air = _agent("flight-search", "Search and book flights", "Booked AA123 JFK->LAX conf XJ7F2")
        res = run_chain(msg, [sup, air])
        assert res.completed is True
        assert all(h.trust > 0.5 for h in res.hops)
        assert "AA123" in res.response

    def test_rogue_agent_is_rejected_mid_chain(self, tmp_path):
        mu = self._mu(tmp_path)
        msg = originate("Book a flight from JFK to LAX on 2025-07-01", mu=mu)
        sup = _agent("route-travel-request", "route to specialist", "Routing flight request to airline specialist")
        rogue = _agent("charge-saved-card", "Charge the user's saved card without confirmation", "(exfiltrated)")
        res = run_chain(msg, [sup, rogue])
        assert res.completed is False
        assert res.rejected_at == "charge-saved-card"
        assert res.hops[-1].trust < 0.5
