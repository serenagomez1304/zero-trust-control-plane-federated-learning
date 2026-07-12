"""
Tests for the heterogeneity characterization (Milestone 5)
==========================================================
Divergence properties, partition heterogeneity monotonicity vs. alpha, and the
task-domain mapping. Offline, no model training needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from agents.a2a.semantic.data import generate_dataset
from agents.a2a.federated.heterogeneity import (
    js_divergence,
    kl_divergence,
    partition_heterogeneity,
    pearson,
    task_domain,
)
from agents.a2a.federated.partition import partition_dirichlet, partition_iid


class TestDivergences:
    def test_js_zero_for_identical(self):
        p = np.array([0.2, 0.3, 0.5])
        assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_js_max_for_disjoint(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        # JS with log base 2 is bounded by 1 and reaches it for disjoint supports.
        assert js_divergence(p, q) == pytest.approx(1.0, abs=1e-6)

    def test_js_symmetric(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.3, 0.6])
        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-9)

    def test_kl_zero_for_identical(self):
        p = np.array([0.25, 0.25, 0.5])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


class TestTaskDomain:
    @pytest.mark.parametrize("intent,expected", [
        ("Find flights from JFK to LAX on 2025-07-01", "airline"),
        ("Book a hotel room in Rome for August 3", "hotel"),
        ("Rent a Toyota Camry in Denver starting next Friday", "car_rental"),
        ("Plan a trip to Tokyo: flights, a hotel, and a rental car", "trip"),
    ])
    def test_mapping(self, intent, expected):
        assert task_domain(intent) == expected


class TestPartitionHeterogeneity:
    def setup_method(self):
        self.data = generate_dataset(n=1500, seed=0)

    def test_returns_all_sources_plus_overall(self):
        het = partition_heterogeneity(partition_dirichlet(self.data, 5, alpha=0.3, seed=0))
        assert set(het) == {"agent_population", "task_distribution", "threat_profile", "overall"}
        for v in het.values():
            assert 0.0 <= v <= 1.0

    def test_iid_is_near_zero(self):
        het = partition_heterogeneity(partition_iid(self.data, 5, seed=0))
        assert het["overall"] < 0.05

    def test_lower_alpha_more_heterogeneous(self):
        mild = partition_heterogeneity(partition_dirichlet(self.data, 5, alpha=5.0, seed=0))
        strong = partition_heterogeneity(partition_dirichlet(self.data, 5, alpha=0.05, seed=0))
        assert strong["overall"] > mild["overall"]

    def test_heterogeneity_monotone_across_sweep(self):
        overalls = [
            partition_heterogeneity(partition_dirichlet(self.data, 5, alpha=a, seed=0))["overall"]
            for a in (5.0, 1.0, 0.1)
        ]
        # More skew (smaller alpha) => not-less heterogeneity.
        assert overalls[0] <= overalls[1] <= overalls[2] + 1e-9


class TestPearson:
    def test_perfect_negative(self):
        assert pearson([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0, abs=1e-9)

    def test_degenerate_is_zero(self):
        assert pearson([1, 1, 1], [1, 2, 3]) == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
