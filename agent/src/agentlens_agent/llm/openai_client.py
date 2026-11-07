"""OpenAI adapter. Translates neutral Message/ToolSpec model to the OpenAI Chat Completions API."""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Sequence
from typing import Any

from agentlens_agent.events import Usage
from agentlens_agent.llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec


def clean_schema_for_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Dereferences $defs/$ref and ensures parameters schema compatibility with OpenAI."""
    s = copy.deepcopy(schema)
    defs = s.pop("$defs", {})
    if not defs:
        defs = s.pop("definitions", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"].split("/")[-1]
                if ref_path in defs:
                    resolved = copy.deepcopy(defs[ref_path])
                    return resolve(resolved)
            cleaned: dict[str, Any] = {}
            for k, v in node.items():
                if k in ("title", "$defs", "definitions", "$schema"):
                    continue
                cleaned[k] = resolve(v)
            return cleaned
        elif isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    resolved_schema = resolve(s)
    if "type" not in resolved_schema:
        resolved_schema["type"] = "object"
    return resolved_schema


def to_openai_messages(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text or ""})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant"}
            if m.text:
                msg["content"] = m.text
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": json.dumps(c.arguments),
                        },
                    }
                    for c in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            for r in m.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": r.tool_call_id,
                        "content": r.content,
                    }
                )
    return out


def parse_openai_response(resp: Any, latency_ms: float) -> LLMResponse:
    choice = resp.choices[0]
    msg = choice.message
    text = msg.content or None
    tool_calls: list[ToolCall] = []

    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception:
                args = {"raw": tc.function.arguments}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                )
            )

    u = getattr(resp, "usage", None)
    cached_tokens = 0
    if u and hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
        cached_tokens = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0

    usage = Usage(
        input_tokens=getattr(u, "prompt_tokens", 0) or 0 if u else 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0 if u else 0,
        cache_read_tokens=cached_tokens,
    )

    return LLMResponse(
        text=text,
        tool_calls=tuple(tool_calls),
        usage=usage,
        stop_reason=str(choice.finish_reason or "end_turn"),
        latency_ms=latency_ms,
    )


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str, api_key: str | None = None, client: Any | None = None) -> None:
        import openai

        self.model = model
        self._openai = openai
        key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = client or openai.AsyncOpenAI(api_key=key)

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        request_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": clean_schema_for_openai(t.input_schema),
                },
            }
            for t in tools
        ]

        formatted_messages = to_openai_messages(system, messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }
        if request_tools:
            kwargs["tools"] = request_tools

        # OpenAI o1/o3-mini handle max_completion_tokens vs max_tokens and may not support temperature
        is_reasoning_model = self.model.startswith("o1") or self.model.startswith("o3")
        if is_reasoning_model:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            retryable = False
            # Check for rate limit or connection errors
            if (
                isinstance(e, (self._openai.RateLimitError, self._openai.APIConnectionError))
                or "rate limit" in msg.lower()
                or "429" in msg
                or "overloaded" in msg.lower()
            ):
                retryable = True
            raise LLMError(f"OpenAI error: {e}", retryable=retryable) from e

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return parse_openai_response(resp, latency_ms)
