#!/usr/bin/env python3
"""
ZTA Benchmark Script
=====================
Measures latency with and without ZTA sidecars.
Runs identical queries through direct ports and sidecar ports,
then reports p50, p95, p99, and mean latencies.

Usage:
    python benchmarks/latency_benchmark.py
    python benchmarks/latency_benchmark.py --runs 50
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import List

import httpx


@dataclass
class BenchmarkResult:
    name: str
    latencies_ms: List[float]
    successes: int
    failures: int

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0

    def _percentile(self, pct: int) -> float:
        if not self.latencies_ms:
            return 0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * pct / 100)
        return sorted_l[min(idx, len(sorted_l) - 1)]


# =========================================================================
# Test cases
# =========================================================================

TESTS = {
    "flight_search": {
        "direct_url": "http://localhost:8080/chat",
        "payload": {"message": "Find flights from JFK to LAX on 2025-08-01"},
    },
    "hotel_search": {
        "direct_url": "http://localhost:8080/chat",
        "payload": {"message": "Search for hotels in New York from August 1 to August 5"},
    },
    "car_rental_search": {
        "direct_url": "http://localhost:8080/chat",
        "payload": {"message": "Find rental cars at LAX from August 1 to August 5"},
    },
}

# A2A direct to agent vs through sidecar
A2A_TESTS = {
    "a2a_airline_direct": {
        "url": "http://localhost:8091/a2a",
        "headers": {"x-agent-id": "supervisor-agent"},
        "payload": {
            "jsonrpc": "2.0", "id": "bench",
            "method": "tasks/send",
            "params": {
                "id": "bench-{run}",
                "message": {"role": "user", "parts": [{"type": "text", "text": "List all airports"}]}
            }
        },
    },
    "a2a_airline_sidecar": {
        "url": "http://localhost:19091/a2a",
        "headers": {"x-agent-id": "supervisor-agent"},
        "payload": {
            "jsonrpc": "2.0", "id": "bench",
            "method": "tasks/send",
            "params": {
                "id": "bench-sc-{run}",
                "message": {"role": "user", "parts": [{"type": "text", "text": "List all airports"}]}
            }
        },
    },
}


async def run_single(client: httpx.AsyncClient, url: str, payload: dict,
                     headers: dict = None, run_id: int = 0) -> tuple[float, bool]:
    """Run a single request and return (latency_ms, success)."""
    # Replace {run} placeholders in payload
    payload_str = json.dumps(payload).replace("{run}", str(run_id))
    payload_data = json.loads(payload_str)

    start = time.perf_counter()
    try:
        resp = await client.post(url, json=payload_data, headers=headers or {}, timeout=30.0)
        elapsed = (time.perf_counter() - start) * 1000
        success = resp.status_code == 200
        if success:
            data = resp.json()
            # /chat endpoint returns {"success": true/false}
            if "success" in data:
                success = data["success"]
            # A2A returns {"result": {"status": {"state": "completed"}}}
            elif "result" in data:
                state = data.get("result", {}).get("status", {}).get("state", "")
                success = state == "completed"
            # If we got a 200 with a message, count it
            elif "message" in data:
                success = True
        return elapsed, success
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, False


async def run_benchmark(name: str, url: str, payload: dict, headers: dict = None,
                        runs: int = 20) -> BenchmarkResult:
    """Run a benchmark suite."""
    latencies = []
    successes = 0
    failures = 0

    async with httpx.AsyncClient() as client:
        # Warmup
        await run_single(client, url, payload, headers, run_id=0)

        for i in range(1, runs + 1):
            latency, success = await run_single(client, url, payload, headers, run_id=i)
            latencies.append(latency)
            if success:
                successes += 1
            else:
                failures += 1

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.2)

    return BenchmarkResult(name=name, latencies_ms=latencies, successes=successes, failures=failures)


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a table."""
    print()
    print("=" * 90)
    print(f"{'Benchmark':<30} {'Runs':>5} {'OK':>4} {'Fail':>4} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print("=" * 90)
    for r in results:
        print(f"{r.name:<30} {len(r.latencies_ms):>5} {r.successes:>4} {r.failures:>4} "
              f"{r.mean:>7.1f}ms {r.p50:>7.1f}ms {r.p95:>7.1f}ms {r.p99:>7.1f}ms")
    print("=" * 90)

    # Compare direct vs sidecar if both exist
    direct = [r for r in results if "direct" in r.name]
    sidecar = [r for r in results if "sidecar" in r.name]
    if direct and sidecar:
        print()
        print("ZTA Overhead (sidecar - direct):")
        for d, s in zip(direct, sidecar):
            overhead = s.mean - d.mean
            pct = (overhead / d.mean * 100) if d.mean > 0 else 0
            print(f"  {d.name} → {s.name}: +{overhead:.1f}ms ({pct:.1f}%)")


async def main():
    parser = argparse.ArgumentParser(description="ZTA Latency Benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per test")
    parser.add_argument("--suite", choices=["e2e", "a2a", "all"], default="all", help="Which tests to run")
    args = parser.parse_args()

    results = []

    if args.suite in ("e2e", "all"):
        print(f"\n--- End-to-End Benchmarks (via /chat, {args.runs} runs each) ---")
        for name, cfg in TESTS.items():
            print(f"  Running {name}...", flush=True)
            r = await run_benchmark(name, cfg["direct_url"], cfg["payload"], runs=args.runs)
            results.append(r)

    if args.suite in ("a2a", "all"):
        print(f"\n--- A2A Direct vs Sidecar ({args.runs} runs each) ---")
        for name, cfg in A2A_TESTS.items():
            print(f"  Running {name}...", flush=True)
            r = await run_benchmark(name, cfg["url"], cfg["payload"], cfg.get("headers"), runs=args.runs)
            results.append(r)

    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())