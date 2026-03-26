#!/usr/bin/env python3
"""
ZTA Adaptive Autoscaler
========================
Monitors multi-signal metrics from the ZTA control plane components and
scales containers up/down based on configurable policies.

Signals: QPS, p95/p99 latency, error rate, CPU/memory.
Threat-response: detects anomalous patterns (QPS spikes, error bursts)
and triggers defensive scaling.

Runs as a standalone service alongside docker-compose.

Usage:
    python services/autoscaler/autoscaler.py
    python services/autoscaler/autoscaler.py --policies services/autoscaler/policies.yaml --dry-run
"""

import argparse
import asyncio
import logging
import os
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("zta-autoscaler")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ComponentMetrics:
    """Collected metrics for a single component."""
    timestamp: float = 0.0
    qps: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    healthy: bool = True
    current_replicas: int = 1


@dataclass
class ScalingDecision:
    """Result of evaluating scaling policies."""
    component: str
    action: str  # "scale_up", "scale_down", "threat_response", "none"
    current_replicas: int
    target_replicas: int
    reason: str
    metrics: ComponentMetrics
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """Collects metrics from Docker stats API and sidecar endpoints."""

    def __init__(self):
        self._request_log: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._latency_log: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._error_log: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._previous_qps: Dict[str, float] = {}

    async def collect(self, component_name: str, config: dict) -> ComponentMetrics:
        """Collect all signals for a component."""
        metrics = ComponentMetrics(timestamp=time.time())

        # Docker stats for CPU/memory
        try:
            stats = await self._get_docker_stats(config["service_name"])
            metrics.cpu_percent = stats.get("cpu_percent", 0.0)
            metrics.memory_mb = stats.get("memory_mb", 0.0)
            metrics.current_replicas = stats.get("replicas", 1)
        except Exception as e:
            logger.debug(f"Docker stats unavailable for {component_name}: {e}")

        # Health check
        health_url = config.get("direct_url") or config.get("metrics_url")
        if health_url:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    start = time.perf_counter()
                    resp = await client.get(health_url)
                    latency = (time.perf_counter() - start) * 1000
                    metrics.healthy = resp.status_code == 200
                    # Use health check latency as a proxy for p95
                    self._latency_log[component_name].append(latency)
            except Exception:
                metrics.healthy = False

        # Calculate derived metrics from history
        latencies = list(self._latency_log[component_name])
        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            metrics.p95_latency_ms = sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[0]
            metrics.p99_latency_ms = sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[0]

        # QPS estimation from Docker logs (simplified — count requests in window)
        try:
            qps = await self._estimate_qps(config["service_name"])
            metrics.qps = qps
        except Exception:
            pass

        # Error rate from sidecar logs
        try:
            error_rate = await self._estimate_error_rate(config["service_name"])
            metrics.error_rate = error_rate
        except Exception:
            pass

        return metrics

    async def _get_docker_stats(self, service_name: str) -> dict:
        """Get CPU/memory stats from docker stats command."""
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format",
                 "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
                capture_output=True, text=True, timeout=10,
            )
            stats = {"cpu_percent": 0.0, "memory_mb": 0.0, "replicas": 0}
            for line in result.stdout.strip().split("\n"):
                if service_name in line:
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        cpu_str = parts[1].replace("%", "").strip()
                        stats["cpu_percent"] = max(stats["cpu_percent"], float(cpu_str))
                        mem_str = parts[2].split("/")[0].strip()
                        if "GiB" in mem_str:
                            stats["memory_mb"] = float(mem_str.replace("GiB", "").strip()) * 1024
                        elif "MiB" in mem_str:
                            stats["memory_mb"] = float(mem_str.replace("MiB", "").strip())
                        stats["replicas"] += 1
            return stats
        except Exception as e:
            return {"cpu_percent": 0.0, "memory_mb": 0.0, "replicas": 1}

    async def _estimate_qps(self, service_name: str) -> float:
        """Estimate QPS from docker logs (count HTTP request lines in last 15s)."""
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", "docker-compose.zta.yml",
                 "logs", "--since", "15s", "--no-log-prefix", service_name],
                capture_output=True, text=True, timeout=10,
            )
            http_lines = [l for l in result.stdout.split("\n")
                          if "HTTP" in l and ("200" in l or "400" in l or "500" in l)]
            return len(http_lines) / 15.0
        except Exception:
            return 0.0

    async def _estimate_error_rate(self, service_name: str) -> float:
        """Estimate error rate from recent logs."""
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", "docker-compose.zta.yml",
                 "logs", "--since", "30s", "--no-log-prefix", service_name],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l for l in result.stdout.split("\n") if "HTTP" in l]
            if not lines:
                return 0.0
            errors = sum(1 for l in lines if any(c in l for c in ["400", "401", "403", "500", "502", "503"]))
            return errors / len(lines)
        except Exception:
            return 0.0

    def get_previous_qps(self, component: str) -> float:
        return self._previous_qps.get(component, 0.0)

    def set_previous_qps(self, component: str, qps: float):
        self._previous_qps[component] = qps


# =============================================================================
# Policy Evaluator
# =============================================================================

class PolicyEvaluator:
    """Evaluates scaling policies against collected metrics."""

    @staticmethod
    def evaluate_condition(signal_value: float, operator: str, threshold: float) -> bool:
        if operator == ">":
            return signal_value > threshold
        elif operator == "<":
            return signal_value < threshold
        elif operator == ">=":
            return signal_value >= threshold
        elif operator == "<=":
            return signal_value <= threshold
        elif operator == "==":
            return signal_value == threshold
        return False

    def evaluate(
        self,
        component_name: str,
        config: dict,
        metrics: ComponentMetrics,
        previous_qps: float,
    ) -> ScalingDecision:
        """Evaluate scaling policies and return a decision."""
        current = metrics.current_replicas or 1
        min_r = config.get("min_replicas", 1)
        max_r = config.get("max_replicas", 5)

        # 1. Check threat-response triggers first (highest priority)
        threat = config.get("threat_response", {})
        if threat:
            # QPS spike detection
            spike_mult = threat.get("qps_spike_multiplier", 3.0)
            if previous_qps > 0 and metrics.qps > previous_qps * spike_mult:
                target = min(threat.get("scale_to_on_threat", max_r), max_r)
                return ScalingDecision(
                    component=component_name, action="threat_response",
                    current_replicas=current, target_replicas=target,
                    reason=f"QPS spike: {previous_qps:.1f} → {metrics.qps:.1f} ({metrics.qps/previous_qps:.1f}x)",
                    metrics=metrics,
                )

            # Error burst detection
            error_thresh = threat.get("error_burst_threshold", 0.5)
            if metrics.error_rate > error_thresh:
                target = min(threat.get("scale_to_on_threat", max_r), max_r)
                return ScalingDecision(
                    component=component_name, action="threat_response",
                    current_replicas=current, target_replicas=target,
                    reason=f"Error burst: {metrics.error_rate:.1%} > {error_thresh:.1%}",
                    metrics=metrics,
                )

        # 2. Check scale-up triggers (ANY condition = scale up)
        scale_up_rules = config.get("scale_up", [])
        for rule in scale_up_rules:
            signal_name = rule["signal"]
            signal_value = getattr(metrics, signal_name, 0.0)
            if self.evaluate_condition(signal_value, rule["operator"], rule["threshold"]):
                if current < max_r:
                    return ScalingDecision(
                        component=component_name, action="scale_up",
                        current_replicas=current, target_replicas=min(current + 1, max_r),
                        reason=f"{signal_name}={signal_value:.1f} {rule['operator']} {rule['threshold']}",
                        metrics=metrics,
                    )

        # 3. Check scale-down triggers (ALL conditions = scale down)
        scale_down_rules = config.get("scale_down", [])
        if scale_down_rules and current > min_r:
            all_met = all(
                self.evaluate_condition(
                    getattr(metrics, rule["signal"], 0.0),
                    rule["operator"],
                    rule["threshold"],
                )
                for rule in scale_down_rules
            )
            if all_met:
                return ScalingDecision(
                    component=component_name, action="scale_down",
                    current_replicas=current, target_replicas=max(current - 1, min_r),
                    reason="All scale-down conditions met",
                    metrics=metrics,
                )

        # 4. No action needed
        return ScalingDecision(
            component=component_name, action="none",
            current_replicas=current, target_replicas=current,
            reason="Within normal parameters",
            metrics=metrics,
        )


# =============================================================================
# Scaler — executes scaling decisions
# =============================================================================

class DockerComposeScaler:
    """Executes scaling decisions via docker compose."""

    def __init__(self, compose_file: str, dry_run: bool = False):
        self.compose_file = compose_file
        self.dry_run = dry_run

    def scale(self, service_name: str, replicas: int) -> bool:
        """Scale a service to the target number of replicas."""
        cmd = [
            "docker", "compose", "-f", self.compose_file,
            "up", "-d", "--scale", f"{service_name}={replicas}",
            "--no-recreate", service_name,
        ]

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return True

        try:
            logger.info(f"Scaling {service_name} to {replicas} replicas")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.info(f"Successfully scaled {service_name} to {replicas}")
                return True
            else:
                logger.error(f"Failed to scale {service_name}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Scaling error: {e}")
            return False


# =============================================================================
# Main Autoscaler Loop
# =============================================================================

class AdaptiveAutoscaler:
    """Main autoscaler — collects metrics, evaluates policies, executes scaling."""

    def __init__(self, policies_path: str, dry_run: bool = False):
        with open(policies_path) as f:
            self.config = yaml.safe_load(f)

        self.global_config = self.config.get("global", {})
        self.components = self.config.get("components", {})
        self.interval = self.global_config.get("evaluation_interval", 15)
        self.cooldown = self.global_config.get("cooldown_after_scale", 60)
        compose_file = self.global_config.get("docker_compose_file", "docker-compose.zta.yml")

        self.collector = MetricsCollector()
        self.evaluator = PolicyEvaluator()
        self.scaler = DockerComposeScaler(compose_file, dry_run=dry_run)

        self._last_scale_time: Dict[str, float] = {}
        self._decision_log: List[ScalingDecision] = []

    async def run_once(self):
        """Run one evaluation cycle across all components."""
        now = time.time()

        for name, comp_config in self.components.items():
            # Check cooldown
            last_scale = self._last_scale_time.get(name, 0)
            if now - last_scale < self.cooldown:
                remaining = int(self.cooldown - (now - last_scale))
                logger.debug(f"{name}: cooldown ({remaining}s remaining)")
                continue

            # Collect metrics
            metrics = await self.collector.collect(name, comp_config)
            previous_qps = self.collector.get_previous_qps(name)

            # Evaluate policy
            decision = self.evaluator.evaluate(name, comp_config, metrics, previous_qps)
            self._decision_log.append(decision)

            # Update QPS history
            self.collector.set_previous_qps(name, metrics.qps)

            # Log decision
            if decision.action != "none":
                logger.info(
                    f"⚡ {decision.action.upper()} {name}: "
                    f"{decision.current_replicas} → {decision.target_replicas} "
                    f"({decision.reason})"
                )

                # Execute scaling
                if decision.target_replicas != decision.current_replicas:
                    success = self.scaler.scale(
                        comp_config["service_name"],
                        decision.target_replicas,
                    )
                    if success:
                        self._last_scale_time[name] = now
            else:
                logger.debug(
                    f"  {name}: OK (qps={metrics.qps:.1f}, p95={metrics.p95_latency_ms:.0f}ms, "
                    f"err={metrics.error_rate:.1%}, cpu={metrics.cpu_percent:.0f}%)"
                )

    async def run(self):
        """Run the autoscaler loop continuously."""
        logger.info(f"ZTA Adaptive Autoscaler started")
        logger.info(f"  Monitoring: {list(self.components.keys())}")
        logger.info(f"  Interval: {self.interval}s, Cooldown: {self.cooldown}s")

        while True:
            try:
                await self.run_once()
            except Exception as e:
                logger.error(f"Evaluation cycle error: {e}", exc_info=True)

            await asyncio.sleep(self.interval)

    def print_status(self):
        """Print current status summary."""
        recent = self._decision_log[-len(self.components):]
        print(f"\n{'='*80}")
        print(f"ZTA Autoscaler Status — {datetime.now(timezone.utc).isoformat()}")
        print(f"{'='*80}")
        print(f"{'Component':<25} {'Action':<18} {'Replicas':<12} {'QPS':>6} {'P95':>8} {'Err%':>6} {'CPU%':>6}")
        print(f"{'-'*80}")
        for d in recent:
            m = d.metrics
            print(
                f"{d.component:<25} {d.action:<18} {d.current_replicas}→{d.target_replicas}"
                f"{'':>5} {m.qps:>6.1f} {m.p95_latency_ms:>7.0f}ms {m.error_rate:>5.1%} {m.cpu_percent:>5.0f}%"
            )
        print(f"{'='*80}\n")


# =============================================================================
# CLI
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="ZTA Adaptive Autoscaler")
    parser.add_argument(
        "--policies",
        default="services/autoscaler/policies.yaml",
        help="Path to scaling policies YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log scaling decisions without executing them",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one evaluation cycle and exit",
    )
    args = parser.parse_args()

    autoscaler = AdaptiveAutoscaler(args.policies, dry_run=args.dry_run)

    if args.once:
        await autoscaler.run_once()
        autoscaler.print_status()
    else:
        await autoscaler.run()


if __name__ == "__main__":
    asyncio.run(main())