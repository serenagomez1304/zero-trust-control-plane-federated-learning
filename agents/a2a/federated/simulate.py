"""
Federated Simulation Loop
=========================
Simulates cross-silo FL of the trust head across K deployments and compares it
to centralized training. The encoder is frozen; each client precomputes its
feature matrix once, then trains the head over the FL rounds.

Produces the data behind Table 2: held-out accuracy per aggregation strategy at
a given heterogeneity level, plus a centralized reference.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from agents.a2a.semantic.data import TrustExample
from agents.a2a.semantic.encoders import get_encoder
from agents.a2a.semantic.model import TrustEvalModel, build_trust_head
from agents.a2a.federated.partition import partition_dirichlet, partition_iid
from agents.a2a.federated.strategies import (
    Strategy,
    build_strategy,
    state_to_vec,
    vec_to_state,
)


@dataclass
class RoundMetric:
    round: int
    accuracy: float
    loss: float


@dataclass
class FederatedResult:
    strategy: str
    n_clients: int
    alpha: float
    rounds: int
    final_accuracy: float
    rounds_to_converge: Optional[int]
    history: List[RoundMetric] = field(default_factory=list)


def _to_tensors(model: TrustEvalModel, examples: Sequence[TrustExample]):
    X = torch.from_numpy(model.features_for(examples))
    y = torch.tensor([float(e.label) for e in examples])
    return X, y


def _evaluate(head: nn.Module, template, weights_vec: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
    head.load_state_dict(vec_to_state(weights_vec, template))
    head.eval()
    with torch.no_grad():
        logits = head(X)
        loss = float(nn.BCEWithLogitsLoss()(logits, y).item())
        preds = (torch.sigmoid(logits) >= 0.5).float()
        acc = float((preds == y).float().mean().item())
    return acc, loss


def federated_train(
    train: List[TrustExample],
    test: List[TrustExample],
    strategy_name: str = "fedavg",
    *,
    encoder_spec: str = "hashing:256",
    n_clients: int = 5,
    rounds: int = 20,
    local_epochs: int = 2,
    lr: float = 0.5,
    batch_size: int = 32,
    alpha: float = 0.5,
    iid: bool = False,
    hidden: int = 128,
    seed: int = 0,
    converge_at: float = 0.9,
) -> FederatedResult:
    """Run one federated training simulation and return its metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TrustEvalModel(get_encoder(encoder_spec), hidden=hidden)
    template: "OrderedDict[str, torch.Tensor]" = OrderedDict(model.head.state_dict())
    dim = sum(v.numel() for v in template.values())

    def head_factory() -> nn.Module:
        return build_trust_head(model.in_dim, hidden=hidden)

    # Partition and precompute per-client features (encoder is frozen).
    shards = (partition_iid(train, n_clients, seed) if iid
              else partition_dirichlet(train, n_clients, alpha=alpha, seed=seed))
    client_data = []
    for shard in shards:
        client_data.append(_to_tensors(model, shard) if shard else None)
    X_test, y_test = _to_tensors(model, test)

    strategy: Strategy = build_strategy(strategy_name)
    strategy.on_init(n_clients, dim)

    global_vec = state_to_vec(template)
    history: List[RoundMetric] = []
    rounds_to_converge: Optional[int] = None

    for rnd in range(rounds):
        results = []
        for cid, data in enumerate(client_data):
            if data is None or data[0].shape[0] == 0:
                continue
            X, y = data
            head = head_factory()
            results.append(strategy.local_update(
                head, template, global_vec, X, y,
                lr=lr, epochs=local_epochs, batch_size=batch_size,
                seed=seed * 100000 + rnd * 100 + cid, client_id=cid,
            ))
        if results:
            global_vec = strategy.aggregate(global_vec, results, lr)

        acc, loss = _evaluate(head_factory(), template, global_vec, X_test, y_test)
        history.append(RoundMetric(rnd, acc, loss))
        if rounds_to_converge is None and acc >= converge_at:
            rounds_to_converge = rnd + 1

    return FederatedResult(
        strategy=strategy_name, n_clients=n_clients,
        alpha=(float("inf") if iid else alpha),
        rounds=rounds, final_accuracy=history[-1].accuracy,
        rounds_to_converge=rounds_to_converge, history=history,
    )


def centralized_accuracy(
    train: List[TrustExample],
    test: List[TrustExample],
    *,
    encoder_spec: str = "hashing:256",
    epochs: int = 40,
    hidden: int = 128,
    seed: int = 0,
) -> float:
    """Centralized reference: train the head on the pooled data."""
    model = TrustEvalModel(get_encoder(encoder_spec), hidden=hidden)
    model.fit(train, epochs=epochs, seed=seed)
    return float(model.evaluate(test)["accuracy"])
