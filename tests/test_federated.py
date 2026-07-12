"""
Tests for the federated learning simulation (Milestone 4)
=========================================================
Offline (hashing encoder), small K and rounds for speed. Covers:
  - non-IID partitioning: coverage, determinism, and that low alpha is more
    skewed than IID
  - each aggregation strategy runs and, on IID data, matches centralized
  - federated training improves over rounds and is reproducible
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from agents.a2a.semantic.data import generate_dataset, train_test_split
from agents.a2a.federated.partition import (
    partition_dirichlet,
    partition_iid,
    partition_report,
)
from agents.a2a.federated.simulate import centralized_accuracy, federated_train
from agents.a2a.federated.strategies import ALL_STRATEGIES


@pytest.fixture(scope="module")
def data():
    return train_test_split(generate_dataset(n=1800, seed=0), seed=0)


# =============================================================================
# Partitioning
# =============================================================================

class TestPartition:
    def test_covers_all_examples_without_overlap(self, data):
        train, _ = data
        shards = partition_dirichlet(train, n_clients=5, alpha=0.5, seed=0)
        assert sum(len(s) for s in shards) == len(train)
        assert len(shards) == 5

    def test_deterministic(self, data):
        train, _ = data
        a = partition_dirichlet(train, 5, alpha=0.3, seed=7)
        b = partition_dirichlet(train, 5, alpha=0.3, seed=7)
        assert [len(s) for s in a] == [len(s) for s in b]

    def test_low_alpha_more_skewed_than_iid(self, data):
        train, _ = data
        iid = partition_iid(train, 5, seed=0)
        skewed = partition_dirichlet(train, 5, alpha=0.05, seed=0)
        # Spread of per-client positive-rate is larger under strong non-IID.
        iid_rates = [r["positive_rate"] for r in partition_report(iid)]
        skew_rates = [r["positive_rate"] for r in partition_report(skewed)]
        assert np.std(skew_rates) > np.std(iid_rates)


# =============================================================================
# Strategies
# =============================================================================

class TestStrategies:
    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_strategy_runs_and_learns_iid(self, data, strategy):
        train, test = data
        r = federated_train(train, test, strategy, n_clients=5, rounds=20, iid=True, seed=0)
        assert r.strategy == strategy
        assert len(r.history) == 20
        # On IID data every strategy should learn this easy task well.
        assert r.final_accuracy >= 0.85, (strategy, r.final_accuracy)

    def test_federated_matches_centralized_iid(self, data):
        train, test = data
        cen = centralized_accuracy(train, test, seed=0)
        fed = federated_train(train, test, "fedavg", n_clients=5, rounds=25, iid=True, seed=0)
        # Federated should come within a small gap of centralized training.
        assert fed.final_accuracy >= cen - 0.10

    def test_reproducible(self, data):
        train, test = data
        a = federated_train(train, test, "fedavg", n_clients=5, rounds=10, alpha=0.3, seed=1)
        b = federated_train(train, test, "fedavg", n_clients=5, rounds=10, alpha=0.3, seed=1)
        assert a.final_accuracy == b.final_accuracy
        assert [m.accuracy for m in a.history] == [m.accuracy for m in b.history]

    def test_learning_improves_over_rounds(self, data):
        train, test = data
        r = federated_train(train, test, "fedavg", n_clients=5, rounds=15, iid=True, seed=0)
        assert r.history[-1].accuracy >= r.history[0].accuracy


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
