"""
Per-Hop Overhead Measurement (Milestone 6)
==========================================
Breaks the substrate's per-hop cost down by pipeline stage
(verify -> trust_eval -> transform -> integrate -> resign) and reports throughput,
for a given semantic model. Used to compare the principal-based baseline against
our per-message model (§6.3).

Timing is done per stage in isolation (the stages are pure given their inputs),
so the numbers are the substrate's own overhead and exclude the untrusted
agent's execution.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

from agents.a2a.substrate import ChainAgent, _sign, originate, verify_signature
from agents.a2a.trust import Attestation, DeclaredPurpose, Mu, SemanticModel

STAGES = ["verify", "trust_eval", "transform", "integrate", "sign"]

_DEFAULT_INTENT = "Book a flight from JFK to LAX on 2025-07-01"
_DEFAULT_RESPONSE = "Booked AA123 JFK->LAX conf XJ7F2"


@dataclass
class StageStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float


@dataclass
class OverheadResult:
    config: str
    stages: Dict[str, StageStats]
    total_ms: float
    throughput_hops_per_s: float


def _pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _time(fn: Callable[[], object], iters: int, warmup: int) -> List[float]:
    samples: List[float] = []
    for i in range(iters + warmup):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            samples.append(dt)
    return samples


def default_agent() -> ChainAgent:
    return ChainAgent(
        agent_id="airline-agent",
        declared_purpose=DeclaredPurpose(label="flight-search", description="Search and book flights"),
        attestation=Attestation(agent_id="airline-agent"),
        handler=lambda view: _DEFAULT_RESPONSE,
    )


def measure_overhead(
    config: str,
    model: SemanticModel,
    *,
    iters: int = 500,
    warmup: int = 50,
    intent: str = _DEFAULT_INTENT,
) -> OverheadResult:
    """Measure per-stage latency for one semantic model."""
    agent = default_agent()
    message = originate(intent, mu=Mu())
    content, kappa = message.content, message.kappa
    purpose, attestation = agent.declared_purpose, agent.attestation

    stage_fns: Dict[str, Callable[[], object]] = {
        "verify": lambda: verify_signature(message),
        "trust_eval": lambda: model.trust_eval(purpose, attestation, kappa),
        "transform": lambda: model.transform(content, purpose, kappa),
        "integrate": lambda: model.integrate(_DEFAULT_RESPONSE, kappa, purpose),
        "sign": lambda: _sign(content, message.mu, kappa),
    }

    stages: Dict[str, StageStats] = {}
    for name in STAGES:
        samples = _time(stage_fns[name], iters, warmup)
        stages[name] = StageStats(statistics.mean(samples), _pct(samples, 50), _pct(samples, 95))

    total = sum(s.mean_ms for s in stages.values())
    throughput = (1000.0 / total) if total > 0 else float("inf")
    return OverheadResult(config, stages, total, throughput)
