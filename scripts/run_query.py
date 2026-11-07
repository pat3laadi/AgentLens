#!/usr/bin/env python3
"""Unified Multi-Model Query Runner for AgentLens (Gemini, Claude, GPT, or Model Arena)."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from uuid import uuid4

from agentlens_agent.config import AgentConfig, load_system_prompt
from agentlens_agent.cost import PricingTable
from agentlens_agent.events import RunFinished, SpanFinished
from agentlens_agent.llm.anthropic_client import AnthropicClient
from agentlens_agent.llm.gemini_client import GeminiClient
from agentlens_agent.llm.openai_client import OpenAIClient
from agentlens_agent.loop import AgentRunner
from agentlens_agent.mcp_registry import McpToolRegistry
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("SEC_USER_AGENT"):
    os.environ["SEC_USER_AGENT"] = "AgentLensResearch agentlens@example.com"

DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o-mini",
}


def build_llm_client(provider: str, model: str):
    if provider == "gemini":
        return GeminiClient(model=model)
    elif provider == "anthropic":
        return AnthropicClient(model=model)
    elif provider in ("openai", "gpt"):
        return OpenAIClient(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def execute_agent(provider: str, model: str, question: str, quiet: bool = False):
    config = AgentConfig(
        name=f"{provider}_runner",
        provider=provider,
        model=model,
        system_prompt_id="system_v1",
        temperature=0.0,
        max_steps=16,
        max_cost_usd=0.50,
        tool_allowlist=[
            "resolve_company",
            "list_filings",
            "search_xbrl_concepts",
            "get_financial_facts",
            "get_filing_section",
            "calculate",
        ],
    )

    prompt, _ = load_system_prompt(config.system_prompt_id)
    pricing = PricingTable.from_yaml("config/pricing.yaml")

    servers = {
        "edgar": "stdio:edgar-mcp",
        "calc": "stdio:calc-mcp",
    }
    registry = McpToolRegistry(servers)
    if not quiet:
        print(
            f"\033[90mConnecting to MCP tools for {provider.upper()} ({model})...\033[0m",
            flush=True,
        )
    await registry.connect()

    llm = build_llm_client(provider, model)
    runner = AgentRunner(
        llm=llm,
        tools=registry,
        config=config,
        system_prompt=prompt,
        pricing=pricing,
        run_id=uuid4(),
    )

    final_result = None
    t0 = time.perf_counter()

    try:
        async for event in runner.run(question):
            if isinstance(event, SpanFinished) and not quiet:
                if event.type == "llm":
                    cost_str = f"${event.cost_usd:.6f}" if event.cost_usd else "$0.00"
                    status_str = (
                        f"\033[32m{event.status}\033[0m"
                        if event.status == "ok"
                        else f"\033[31m{event.status}\033[0m"
                    )
                    print(
                        f"[{provider.upper()}] LLM Step: {event.latency_ms:.1f}ms | Cost: {cost_str} | Status: {status_str}",
                        flush=True,
                    )
            elif isinstance(event, RunFinished):
                final_result = event
    finally:
        await registry.close()

    elapsed = time.perf_counter() - t0
    return {
        "provider": provider,
        "model": model,
        "result": final_result,
        "elapsed": elapsed,
    }


async def run_single(provider: str, model: str, question: str):
    print("\n\033[1;36m=== AgentLens Single-Model Run ===\033[0m")
    print(f"Provider: \033[1;32m{provider}\033[0m | Model: \033[1m{model}\033[0m")
    print(f"Question: \033[1m{question}\033[0m\n")

    out = await execute_agent(provider, model, question, quiet=False)
    res = out["result"]

    print("\n" + "=" * 60)
    if res:
        status = getattr(res.status, "value", str(res.status))
        reason = getattr(res.termination_reason, "value", str(res.termination_reason))
        cost = getattr(res.totals, "cost_usd", 0.0)
        steps = getattr(res.totals, "steps", 0)
        tools = getattr(res.totals, "tool_calls", 0)
        print(f"\033[1;32m[Run Complete]\033[0m Status: {status} (Reason: {reason})")
        print(
            f"Steps: {steps} | Tool Calls: {tools} | Cost: ${cost:.5f} | Latency: {out['elapsed']:.2f}s"
        )
        if res.answer:
            ans = res.answer
            print("\n\033[1;33m=== Structured Answer ===\033[0m")
            if ans.value is not None:
                print(f"Value: {ans.value:,.2f} {ans.unit or ''}")
            if ans.period:
                print(f"Period: {ans.period}")
            if ans.text:
                print(f"\nExplanation: {ans.text}")
            if ans.citations:
                print("\nCitations:")
                for c in ans.citations:
                    acc = getattr(c, "accession", None)
                    sec = getattr(c, "section", None)
                    cnc = getattr(c, "concept", None)
                    print(
                        f"  • Accession: {acc or 'N/A'} (Section: {sec or 'N/A'}) (Concept: {cnc or 'N/A'})"
                    )
    else:
        print("\033[1;31mRun finished without result.\033[0m")
    print("=" * 60)


async def run_arena(question: str):
    print("\n\033[1;35m============================================================\033[0m")
    print("\033[1;35m       ⚔️  AgentLens Model Arena: Multi-Model Evaluation  ⚔️\033[0m")
    print("\033[1;35m============================================================\033[0m")
    print(f"Question: \033[1m{question}\033[0m\n")

    # Filter to available providers by checking API keys
    active_providers = []
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        active_providers.append(("gemini", DEFAULT_MODELS["gemini"]))
    else:
        print("\033[90m(Skipping Gemini: GEMINI_API_KEY not configured)\033[0m")

    if os.environ.get("ANTHROPIC_API_KEY"):
        active_providers.append(("anthropic", DEFAULT_MODELS["anthropic"]))
    else:
        print("\033[90m(Skipping Anthropic: ANTHROPIC_API_KEY not configured)\033[0m")

    if os.environ.get("OPENAI_API_KEY"):
        active_providers.append(("openai", DEFAULT_MODELS["openai"]))
    else:
        print("\033[90m(Skipping OpenAI: OPENAI_API_KEY not configured)\033[0m")

    if not active_providers:
        print("\033[31mError: No API keys configured in environment.\033[0m")
        return

    print(f"\nLaunching {len(active_providers)} models simultaneously in parallel...\n")
    tasks = [execute_agent(p, m, question, quiet=False) for p, m in active_providers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 70)
    print("\033[1;36m                 ⚔️  ARENA SCORECARD & LEADERBOARD  ⚔️\033[0m")
    print("=" * 70)
    print(f"{'Provider':<12} | {'Model':<25} | {'Cost ($)':<10} | {'Latency':<8} | {'Status':<10}")
    print("-" * 70)

    for r in results:
        if isinstance(r, Exception):
            print(f"Error: {r}")
            continue
        p = r["provider"].upper()
        m = r["model"][:24]
        res = r["result"]
        cost = getattr(res.totals, "cost_usd", 0.0) if res else 0.0
        status = getattr(res.status, "value", str(res.status)) if res else "failed"
        lat = f"{r['elapsed']:.1f}s"
        print(f"{p:<12} | {m:<25} | ${cost:<9.5f} | {lat:<8} | {status:<10}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AgentLens Multi-Model Runner")
    parser.add_argument("question", nargs="*", help="Question to run")
    parser.add_argument(
        "--provider",
        choices=["gemini", "anthropic", "openai"],
        default="gemini",
        help="LLM Provider",
    )
    parser.add_argument("--model", help="Model name override")
    parser.add_argument(
        "--arena", action="store_true", help="Run all available models simultaneously and compare"
    )

    args = parser.parse_args()
    question = (
        " ".join(args.question).strip()
        if args.question
        else "For fiscal year 2023, which company had higher total revenue: NVIDIA or IBM?"
    )

    provider = args.provider
    model = args.model or DEFAULT_MODELS[provider]

    if args.arena:
        asyncio.run(run_arena(question))
    else:
        asyncio.run(run_single(provider, model, question))


if __name__ == "__main__":
    main()
