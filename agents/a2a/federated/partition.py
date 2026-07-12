"""
Non-IID Partitioning across deployments
=======================================
Split the synthetic trust dataset across K deployments so that each holds a
different distribution — the non-IID setting the paper characterizes (§5.3).

We use the standard Dirichlet(alpha) scheme: examples are grouped by a key, and
each group is split across clients with proportions drawn from Dirichlet(alpha).
Low ``alpha`` → highly skewed (very non-IID); high ``alpha`` → near-uniform (IID).

The default group key is ``(label, kind, domain)``, which simultaneously skews
all three heterogeneity sources the paper names:
  - the **threat profile** (``label``: consistent vs. rogue balance),
  - the **agent-population / attack mix** (``kind``: legit / cross_domain /
    escalation), and
  - the **task distribution** (``domain``: the intent's task domain).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from agents.a2a.semantic.data import TrustExample


def default_group_key(ex: TrustExample) -> str:
    return f"{ex.label}:{ex.kind}:{ex.domain}"


def partition_dirichlet(
    examples: List[TrustExample],
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 0,
    group_key: Callable[[TrustExample], str] = default_group_key,
) -> List[List[TrustExample]]:
    """Partition ``examples`` into ``n_clients`` non-IID shards via Dirichlet(alpha)."""
    rng = np.random.default_rng(seed)

    groups: dict[str, List[TrustExample]] = {}
    for ex in examples:
        groups.setdefault(group_key(ex), []).append(ex)

    clients: List[List[TrustExample]] = [[] for _ in range(n_clients)]
    for _, items in sorted(groups.items()):
        idx = rng.permutation(len(items))
        proportions = rng.dirichlet([alpha] * n_clients)
        # Cut points for splitting this group's (shuffled) items across clients.
        cuts = (np.cumsum(proportions) * len(items)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(idx, cuts)):
            clients[client_id].extend(items[i] for i in chunk)

    for shard in clients:
        rng.shuffle(shard)
    return clients


def partition_iid(
    examples: List[TrustExample],
    n_clients: int,
    seed: int = 0,
) -> List[List[TrustExample]]:
    """Uniform random partition (the IID reference)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples))
    return [[examples[i] for i in shard] for shard in np.array_split(idx, n_clients)]


def partition_report(clients: List[List[TrustExample]]) -> List[dict]:
    """Per-client summary (size, positive rate) — useful for logging/heterogeneity."""
    report = []
    for cid, shard in enumerate(clients):
        n = len(shard)
        pos = sum(e.label for e in shard)
        report.append({
            "client": cid,
            "n": n,
            "positive_rate": (pos / n) if n else 0.0,
        })
    return report
