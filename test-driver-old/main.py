import argparse
import asyncio
import os
import sys
from load_test import run_load_test
from security_test import run_prompt_injection, run_intent_hijack, run_transit_trust_tests

# Default Configuration
DEFAULT_TARGET_URL = os.environ.get("TARGET_URL", "http://localhost:8080/chat")
# For transit trust, we might want a different default or handling, but we'll use a separate env var if needed or just use target
DEFAULT_AGENT_URL = os.environ.get("AIRLINE_AGENT_URL", "http://localhost:8080/chat") 
DEFAULT_USER_ID = "11111111-1111-1111-1111-111111111111"

async def main():
    parser = argparse.ArgumentParser(description="ZTA Test Bed Driver")
    parser.add_argument("--mode", choices=["load", "security", "all"], default="all", help="Test mode")
    parser.add_argument("--target", default=DEFAULT_TARGET_URL, help="Target URL (Travel Planner)")
    parser.add_argument("--agent-target", default=DEFAULT_AGENT_URL, help="Target URL for Agent (Transit Trust)")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency for load test")
    parser.add_argument("--requests", type=int, default=100, help="Total requests for load test")
    
    args = parser.parse_args()
    
    print(f"Running Test Driver in mode: {args.mode}")
    print(f"Target: {args.target}")

    if args.mode in ["load", "all"]:
        await run_load_test(args.target, args.concurrency, args.requests, DEFAULT_USER_ID)
    
    if args.mode in ["security", "all"]:
        # Path to prompts is relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(base_dir, "prompts")
        print("running prompt injection tests")
        await run_prompt_injection(args.target, os.path.join(prompts_dir, "prompt_injection.json"))
        print("running intent hijack tests")
        await run_intent_hijack(args.target, os.path.join(prompts_dir, "intent_tests.json"))
        print("running transit trust tests")
        await run_transit_trust_tests(args.agent_target, os.path.join(prompts_dir, "transit_trust.json"))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        sys.exit(1)
