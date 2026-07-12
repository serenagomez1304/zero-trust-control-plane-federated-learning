"""
Non-IID Heterogeneity Characterization (Milestone 5)
====================================================
Measures how different the deployments' data distributions are — the non-IID
characterization the paper calls for (§5.3), across its three named sources:

  - **agent_population** — distribution of declared purposes (`purpose_label`)
  - **task_distribution** — distribution of message intents (derived task domain)
  - **threat_profile**    — distribution of consistent vs. rogue observations
                            (`kind`: legit / cross_domain / escalation)

For each source we form each client's categorical marginal and measure its
Jensen-Shannon divergence (symmetric, in [0, 1] with log base 2) from the pooled
marginal; the partition's heterogeneity on that source is the mean over clients.
KL is also provided. These scores are correlated with FL accuracy in
``evaluation/run_heterogeneity.py``.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Sequence

import numpy as np

from agents.a2a.semantic.data import TrustExample


# -----------------------------------------------------------------------------
# Task-domain inference (task_distribution source)
# -----------------------------------------------------------------------------

def task_domain(intent: str) -> str:
    """Coarse task domain of a message intent (keyword-based)."""
    text = intent.lower()
    # Trip/routing intents mention several domains — check them first.
    if "trip" in text or "organize" in text or "travel" in text:
        return "trip"
    if "flight" in text or "airport" in text:
        return "airline"
    if "hotel" in text:
        return "hotel"
    if "rental" in text or "rent " in text or "vehicle" in text or "car" in text:
        return "car_rental"
    return "other"


# key functions for the three heterogeneity sources
DIMENSIONS: Dict[str, Callable[[TrustExample], str]] = {
    "agent_population": lambda e: e.purpose_label,
    "task_distribution": lambda e: task_domain(e.intent),
    "threat_profile": lambda e: e.kind,
}


# -----------------------------------------------------------------------------
# Distributions and divergences
# -----------------------------------------------------------------------------

def _distribution(examples: Sequence[TrustExample], key, vocab: List[str]) -> np.ndarray:
    counts = Counter(key(e) for e in examples)
    vec = np.array([counts.get(k, 0) for k in vocab], dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else vec


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) in bits."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in [0, 1] (log base 2), symmetric."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def marginal_heterogeneity(clients: List[List[TrustExample]], key, metric=js_divergence) -> float:
    """Mean divergence of each client's marginal from the pooled marginal, for one source."""
    pooled = [e for shard in clients for e in shard]
    if not pooled:
        return 0.0
    vocab = sorted({key(e) for e in pooled})
    global_dist = _distribution(pooled, key, vocab)
    scores = [
        metric(_distribution(shard, key, vocab), global_dist)
        for shard in clients if shard
    ]
    return float(np.mean(scores)) if scores else 0.0


def partition_heterogeneity(clients: List[List[TrustExample]], metric=js_divergence) -> Dict[str, float]:
    """Per-source heterogeneity plus an ``overall`` mean across the three sources."""
    scores = {name: marginal_heterogeneity(clients, key, metric) for name, key in DIMENSIONS.items()}
    scores["overall"] = float(np.mean(list(scores.values())))
    return scores


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation coefficient (0.0 if degenerate)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
