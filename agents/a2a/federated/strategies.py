"""
Federated Aggregation Strategies
================================
FedAvg, FedProx, SCAFFOLD, FedNova, operating on the trust head's weights (a
small flat vector, since only the head is federated). Each strategy defines how
a client trains locally and how the server aggregates the results.

References:
  FedAvg   — McMahan et al. 2017
  FedProx  — Li et al. 2020 (proximal term for client drift)
  SCAFFOLD — Karimireddy et al. 2020 (control variates)
  FedNova  — Wang et al. 2020 (normalized averaging for heterogeneous local steps)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# --- weight <-> flat-vector helpers ------------------------------------------

def state_to_vec(state: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
    return torch.cat([v.flatten() for v in state.values()])


def vec_to_state(vec: torch.Tensor, template: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    i = 0
    for k, v in template.items():
        n = v.numel()
        out[k] = vec[i : i + n].view_as(v).clone()
        i += n
    return out


@dataclass
class ClientResult:
    weights: torch.Tensor   # local weights y_i (flat)
    n: int                  # number of local samples
    steps: int              # number of local SGD steps (tau_i)


def _run_local_sgd(
    head: nn.Module,
    template: "OrderedDict[str, torch.Tensor]",
    global_vec: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    prox_mu: float = 0.0,
    correction: Optional[torch.Tensor] = None,  # SCAFFOLD (c_global - c_local), flat
) -> int:
    """Train ``head`` in place from ``global_vec``; return the number of steps."""
    torch.manual_seed(seed)
    head.load_state_dict(vec_to_state(global_vec, template))
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    global_named: Dict[str, torch.Tensor] = vec_to_state(global_vec, template) if prox_mu > 0 else {}
    corr_named: Dict[str, torch.Tensor] = vec_to_state(correction, template) if correction is not None else {}

    n = X.shape[0]
    steps = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(head(X[idx]), y[idx])
            if prox_mu > 0:
                prox = sum(((p - global_named[name]) ** 2).sum() for name, p in head.named_parameters())
                loss = loss + (prox_mu / 2.0) * prox
            loss.backward()
            if corr_named:
                # SCAFFOLD: g <- g + (c_global - c_local)
                for name, p in head.named_parameters():
                    if p.grad is not None:
                        p.grad.add_(corr_named[name])
            opt.step()
            steps += 1
    return steps


def _weighted_average(results: List[ClientResult]) -> torch.Tensor:
    total = sum(r.n for r in results)
    return sum((r.n / total) * r.weights for r in results)


class Strategy(ABC):
    name: str = "base"

    def on_init(self, n_clients: int, dim: int) -> None:
        """Hook for stateful strategies (e.g. SCAFFOLD control variates)."""

    @abstractmethod
    def local_update(self, head, template, global_vec, X, y, lr, epochs, batch_size, seed, client_id) -> ClientResult:
        ...

    @abstractmethod
    def aggregate(self, global_vec: torch.Tensor, results: List[ClientResult], lr: float) -> torch.Tensor:
        ...


class FedAvg(Strategy):
    name = "fedavg"

    def local_update(self, head, template, global_vec, X, y, lr, epochs, batch_size, seed, client_id):
        steps = _run_local_sgd(head, template, global_vec, X, y, lr, epochs, batch_size, seed)
        return ClientResult(state_to_vec(head.state_dict()), X.shape[0], steps)

    def aggregate(self, global_vec, results, lr):
        return _weighted_average(results)


class FedProx(Strategy):
    name = "fedprox"

    def __init__(self, mu: float = 0.1):
        self.mu = mu

    def local_update(self, head, template, global_vec, X, y, lr, epochs, batch_size, seed, client_id):
        steps = _run_local_sgd(head, template, global_vec, X, y, lr, epochs, batch_size, seed, prox_mu=self.mu)
        return ClientResult(state_to_vec(head.state_dict()), X.shape[0], steps)

    def aggregate(self, global_vec, results, lr):
        return _weighted_average(results)


class FedNova(Strategy):
    """Normalized averaging: accounts for clients running different #local steps."""
    name = "fednova"

    def local_update(self, head, template, global_vec, X, y, lr, epochs, batch_size, seed, client_id):
        steps = _run_local_sgd(head, template, global_vec, X, y, lr, epochs, batch_size, seed)
        return ClientResult(state_to_vec(head.state_dict()), X.shape[0], steps)

    def aggregate(self, global_vec, results, lr):
        total = sum(r.n for r in results)
        # Normalized local update d_i = (x - y_i)/tau_i ; effective steps tau_eff = sum p_i tau_i.
        tau_eff = sum((r.n / total) * r.steps for r in results)
        agg_d = sum((r.n / total) * ((global_vec - r.weights) / max(r.steps, 1)) for r in results)
        return global_vec - tau_eff * agg_d


class Scaffold(Strategy):
    """SCAFFOLD with control variates (option II update), server lr = 1."""
    name = "scaffold"

    def __init__(self):
        self.c_global: Optional[torch.Tensor] = None
        self.c_clients: List[torch.Tensor] = []
        self._deltas: Dict[int, torch.Tensor] = {}

    def on_init(self, n_clients: int, dim: int) -> None:
        self.c_global = torch.zeros(dim)
        self.c_clients = [torch.zeros(dim) for _ in range(n_clients)]
        self._deltas = {}

    def local_update(self, head, template, global_vec, X, y, lr, epochs, batch_size, seed, client_id):
        c_local = self.c_clients[client_id]
        correction = self.c_global - c_local
        steps = _run_local_sgd(head, template, global_vec, X, y, lr, epochs, batch_size, seed, correction=correction)
        local_vec = state_to_vec(head.state_dict())

        # c_i^+ = c_i - c + (x - y_i) / (steps * lr)
        new_c_local = c_local - self.c_global + (global_vec - local_vec) / max(steps * lr, 1e-12)
        self._deltas[client_id] = new_c_local - c_local
        self.c_clients[client_id] = new_c_local
        return ClientResult(local_vec, X.shape[0], steps)

    def aggregate(self, global_vec, results, lr):
        # c^+ = c + (1/N) sum Δc_i  ;  x^+ = x + (1/|S|) sum (y_i - x)
        n_clients = len(self.c_clients)
        if self._deltas:
            self.c_global = self.c_global + sum(self._deltas.values()) / n_clients
            self._deltas = {}
        delta_avg = sum((r.weights - global_vec) for r in results) / len(results)
        return global_vec + delta_avg


def build_strategy(name: str, **kwargs) -> Strategy:
    name = name.lower()
    if name == "fedavg":
        return FedAvg()
    if name == "fedprox":
        return FedProx(mu=kwargs.get("mu", 0.1))
    if name == "scaffold":
        return Scaffold()
    if name == "fednova":
        return FedNova()
    raise ValueError(f"unknown strategy: {name!r}")


ALL_STRATEGIES = ["fedavg", "fedprox", "scaffold", "fednova"]
