#!/usr/bin/env python3
"""Interactive query runner for AgentLens using Gemini."""

from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

from agentlens_agent.config import AgentConfig, load_system_prompt
from agentlens_agent.cost import PricingTable
from agentlens_agent.events import RunFinished, SpanFinished
from agentlens_agent.llm.gemini_client import GeminiClient
from agentlens_agent.loop import AgentRunner
from agentlens_agent.mcp_registry import McpToolRegistry
from dotenv import load_dotenv

load_dotenv()

# Set default SEC user agent if not provided
if not os.environ.get("SEC_USER_AGENT"):
    os.environ["SEC_USER_AGENT"] = "AgentLensResearch agentlens@example.com"


def prompt_for_inputs() -> tuple[str, str]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not api_key:
        print("\033[93mNotice: GEMINI_API_KEY is not set in environment or .env file.\033[0m")
        print("Please export GEMINI_API_KEY=your_key or enter it below:")
        try:
            api_key = input("Enter Gemini API Key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(1)
        if not api_key:
            print("Error: Gemini API Key is required to run live queries.")
            sys.exit(1)
        os.environ["GEMINI_API_KEY"] = api_key

    # Determine question
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("\n\033[1;36m=== AgentLens SEC Financial Agent (Powered by Gemini) ===\033[0m")
        print("Example questions:")
        print("  1. What was Apple's total net sales for fiscal year 2023?")
        print("  2. By what percentage did Microsoft's revenue change from fiscal 2022 to 2023?")
        print(
            "  3. For fiscal 2023, which company had higher operating margin: Alphabet or Microsoft?"
        )
        print()
        try:
            question = input("\033[1mEnter your question: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

    if not question:
        question = "What was Apple's total net sales for fiscal year 2023?"
        print(f"Using default question: {question}")

    return api_key, question


async def run_query(api_key: str, question: str) -> None:
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    print(f"\n\033[1;34m[AgentLens]\033[0m Initializing agent with \033[1m{model_name}\033[0m...")
    print(f"\033[1;34m[AgentLens]\033[0m Question: \033[1m{question}\033[0m\n")

    config = AgentConfig(
        name="gemini_runner",
        provider="gemini",
        model=model_name,
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

    prompt, _sha = load_system_prompt(config.system_prompt_id)
    pricing = PricingTable.from_yaml("config/pricing.yaml")

    # Connect to MCP tool servers (edgar-mcp and calc-mcp)
    servers = {
        "edgar": "stdio:edgar-mcp",
        "calc": "stdio:calc-mcp",
    }
    registry = McpToolRegistry(servers)

    print("\033[90mConnecting to MCP tools (SEC EDGAR & Safe Calculator)...\033[0m", flush=True)
    await registry.connect()
    print("\033[32m✔ MCP tools connected.\033[0m\n" + "-" * 60, flush=True)

    runner = AgentRunner(
        llm=GeminiClient(model=model_name, api_key=api_key),
        tools=registry,
        config=config,
        system_prompt=prompt,
        pricing=pricing,
        run_id=uuid4(),
    )

    final_result = None

    try:
        async for event in runner.run(question):
            if isinstance(event, SpanFinished):
                duration_ms = (event.ended_at - event.started_at).total_seconds() * 1000
                cost = event.cost_usd or 0.0
                if event.type == "llm_call":
                    print(
                        f"\n\033[1;35m[LLM Step]\033[0m Completed in {duration_ms:.1f}ms | Cost: ${cost:.6f} | Status: {event.status}",
                        flush=True,
                    )
                    if event.error:
                        print(f"\033[91mError:\033[0m {event.error}", flush=True)
                    if event.output and event.output.get("text"):
                        print(f"\033[90mThought:\033[0m {event.output['text']}", flush=True)
                    if event.output and event.output.get("tool_calls"):
                        for tc in event.output["tool_calls"]:
                            print(
                                f"  \033[96m→ Model called tool:\033[0m {tc['name']}({tc.get('arguments', {})})",
                                flush=True,
                            )
                elif event.type == "tool_call":
                    tool_name = event.name
                    print(f"  \033[1;33m🛠  Tool: {tool_name}\033[0m", flush=True)
                    print(f"     \033[90mArgs:\033[0m {event.input}", flush=True)
                    if event.output and "content" in event.output:
                        preview = str(event.output["content"])
                        if len(preview) > 200:
                            preview = preview[:200] + "..."
                        print(f"     \033[90mResult:\033[0m {preview}", flush=True)
            elif isinstance(event, RunFinished):
                final_result = event

        print("\n" + "=" * 60, flush=True)
        if final_result:
            status_str = getattr(final_result.status, "value", str(final_result.status))
            reason_str = getattr(
                final_result.termination_reason, "value", str(final_result.termination_reason)
            )
            print(
                f"\033[1;32m[Run Complete]\033[0m Status: \033[1m{status_str}\033[0m (Reason: {reason_str})",
                flush=True,
            )
            if final_result.error:
                print(f"\033[91mFailure Error:\033[0m {final_result.error}", flush=True)
            latency_s = getattr(final_result.totals, "latency_ms", 0) / 1000
            print(
                f"Total Steps: {final_result.totals.steps} | Tool Calls: {final_result.totals.tool_calls} | "
                f"Total Cost: ${final_result.totals.cost_usd:.5f} | Latency: {latency_s:.2f}s",
                flush=True,
            )

            ans = final_result.final_answer
            if ans:
                print("\n\033[1;36m=== Structured Answer ===\033[0m", flush=True)
                if ans.abstained:
                    print(f"\033[1;33mAbstained:\033[0m Yes - {ans.text}", flush=True)
                else:
                    if ans.value is not None:
                        print(f"\033[1mValue:\033[0m {ans.value:,.2f}", flush=True)
                    if ans.unit:
                        print(f"\033[1mUnit:\033[0m {ans.unit}", flush=True)
                    if ans.period:
                        print(f"\033[1mPeriod:\033[0m {ans.period}", flush=True)
                    print(f"\033[1mExplanation:\033[0m {ans.text}", flush=True)
                    if ans.citations:
                        print("\n\033[1mCitations:\033[0m", flush=True)
                        for c in ans.citations:
                            acc = c.accession
                            sec = f" (Section: {c.section})" if c.section else ""
                            concept = f" (Concept: {c.concept})" if c.concept else ""
                            print(f"  • SEC Accession: {acc}{sec}{concept}", flush=True)
        print("=" * 60 + "\n", flush=True)
    finally:
        await registry.close()


def main() -> None:
    api_key, question = prompt_for_inputs()
    asyncio.run(run_query(api_key, question))


if __name__ == "__main__":
    main()
