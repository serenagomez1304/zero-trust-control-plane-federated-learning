"""
Tests for the per-hop overhead measurement (Milestone 6)
========================================================
Functional checks (small iters), not timing assertions beyond the robust
"learned trust_eval costs more than the baseline's identity check".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agents.a2a.semantic.data import generate_dataset, train_test_split
from agents.a2a.semantic.encoders import get_encoder
from agents.a2a.semantic.model import RealSemanticModel, TrustEvalModel
from agents.a2a.trust import StubSemanticModel
from evaluation.attacks import PrincipalBaselineModel
from evaluation.overhead import STAGES, measure_overhead


@pytest.fixture(scope="module")
def ours_model():
    train, _ = train_test_split(generate_dataset(n=800, seed=0), seed=0)
    model = TrustEvalModel(get_encoder("hashing:256"))
    model.fit(train, epochs=20, seed=0)
    return RealSemanticModel(model)


class TestOverhead:
    def test_returns_all_stages_and_positive_totals(self):
        r = measure_overhead("stub", StubSemanticModel(), iters=40, warmup=10)
        assert set(r.stages) == set(STAGES)
        assert r.total_ms > 0.0
        assert r.throughput_hops_per_s > 0.0
        assert all(s.mean_ms >= 0.0 for s in r.stages.values())
        # total is the sum of stage means
        assert r.total_ms == pytest.approx(sum(s.mean_ms for s in r.stages.values()), rel=1e-6)

    def test_learned_trust_eval_costs_more_than_baseline(self, ours_model):
        ours = measure_overhead("ours", ours_model, iters=60, warmup=15)
        base = measure_overhead("baseline", PrincipalBaselineModel(), iters=60, warmup=15)
        # The learned trust_eval (encode + head forward) is heavier than the
        # baseline's identity lookup, so ours has the larger trust_eval + total.
        assert ours.stages["trust_eval"].mean_ms > base.stages["trust_eval"].mean_ms
        assert ours.total_ms > base.total_ms


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
