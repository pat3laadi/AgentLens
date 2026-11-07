"""Gemini adapter. Translates our neutral Message/ToolSpec model to the Google GenAI SDK."""

from __future__ import annotations

import asyncio
import copy
import os
import time
import uuid
from collections.abc import Sequence
from typing import Any

from agentlens_agent.events import Usage
from agentlens_agent.llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec


def clean_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Dereferences $defs/$ref and adjusts OpenAPI/JSON Schema for Gemini."""
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
            # handle anyOf: [{"type": "..."}, {"type": "null"}]
            if "anyOf" in node:
                non_nulls = [x for x in node["anyOf"] if x != {"type": "null"}]
                has_null = len(non_nulls) < len(node["anyOf"])
                if len(non_nulls) == 1:
                    base = resolve(non_nulls[0])
                    if has_null and isinstance(base, dict):
                        base["nullable"] = True
                    return base
            cleaned: dict[str, Any] = {}
            for k, v in node.items():
                if k in ("title", "$defs", "definitions", "$schema"):
                    continue
                cleaned[k] = resolve(v)
            return cleaned
        elif isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(s)


def to_gemini_contents(
    messages: Sequence[Message],
    thought_signatures: dict[str, bytes] | None = None,
) -> list[Any]:
    from google.genai import types

    ts_map = thought_signatures or {}
    contents: list[types.Content] = []
    for m in messages:
        if m.role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=m.text or "")],
                )
            )
        elif m.role == "assistant":
            parts: list[types.Part] = []
            if m.text:
                parts.append(types.Part.from_text(text=m.text))
            for c in m.tool_calls:
                p = types.Part.from_function_call(name=c.name, args=c.arguments)
                if c.id in ts_map:
                    p.thought_signature = ts_map[c.id]
                parts.append(p)
            contents.append(types.Content(role="model", parts=parts))
        elif m.role == "tool":
            parts = []
            for r in m.tool_results:
                # recover tool name from tool_call_id "name:uuid"
                name = r.tool_call_id.split(":", 1)[0]
                res_dict = {"error": r.content} if r.is_error else {"result": r.content}
                parts.append(types.Part.from_function_response(name=name, response=res_dict))
            contents.append(types.Content(role="user", parts=parts))
    return contents


def parse_gemini_response(
    resp: Any,
    latency_ms: float,
    thought_signatures: dict[str, bytes] | None = None,
) -> LLMResponse:
    texts: list[str] = []
    calls: list[ToolCall] = []

    if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
        for part in resp.candidates[0].content.parts:
            if getattr(part, "text", None):
                texts.append(part.text)
            if getattr(part, "function_call", None):
                fc = part.function_call
                call_id = f"{fc.name}:{uuid.uuid4().hex[:8]}"
                args = dict(fc.args) if fc.args else {}
                calls.append(ToolCall(id=call_id, name=fc.name, arguments=args))
                if getattr(part, "thought_signature", None) and thought_signatures is not None:
                    thought_signatures[call_id] = part.thought_signature

    stop_reason = "end_turn"
    if resp.candidates and getattr(resp.candidates[0], "finish_reason", None):
        stop_reason = str(resp.candidates[0].finish_reason)
    if calls:
        stop_reason = "tool_use"

    u = getattr(resp, "usage_metadata", None)
    usage = Usage(
        input_tokens=getattr(u, "prompt_token_count", 0) or 0,
        output_tokens=getattr(u, "candidates_token_count", 0) or 0,
        cache_read_tokens=getattr(u, "cached_content_token_count", 0) or 0,
    )

    return LLMResponse(
        text="\n".join(texts) or None,
        tool_calls=tuple(calls),
        usage=usage,
        stop_reason=stop_reason,
        latency_ms=latency_ms,
    )


class GeminiClient:
    provider = "gemini"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        from google import genai

        self.model = model
        self._genai = genai
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self._client = client or genai.Client(api_key=self.api_key)
        self._thought_signatures: dict[str, bytes] = {}

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        from google.genai import types
        from google.genai.errors import APIError

        t0 = time.perf_counter()

        func_decls = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=clean_schema_for_gemini(t.input_schema),
            )
            for t in tools
        ]

        tool_config = [types.Tool(function_declarations=func_decls)] if func_decls else None

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=tool_config,
        )

        contents = to_gemini_contents(messages, self._thought_signatures)

        # Retry transient 503 / 429 errors
        for attempt in range(3):
            try:
                resp = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return parse_gemini_response(
                    resp,
                    (time.perf_counter() - t0) * 1000,
                    self._thought_signatures,
                )
            except APIError as e:
                code = getattr(e, "code", None) or 500
                retryable = code in (408, 409, 429) or code >= 500
                if retryable and attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise LLMError(f"gemini {code}: {e}", retryable=retryable) from e
            except Exception as e:
                raise LLMError(f"gemini error: {e}", retryable=False) from e

        raise LLMError("gemini failed after retries", retryable=False)
