import asyncio
from types import SimpleNamespace

from agentlens_agent.llm.base import Message, ToolCall, ToolResult
from agentlens_agent.llm.openai_client import (
    OpenAIClient,
    clean_schema_for_openai,
    parse_openai_response,
    to_openai_messages,
)


def test_openai_schema_cleaning():
    raw_schema = {
        "$defs": {
            "Item": {
                "title": "Item",
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        },
        "title": "TopLevel",
        "type": "object",
        "properties": {
            "item": {"$ref": "#/$defs/Item"},
        },
    }
    cleaned = clean_schema_for_openai(raw_schema)
    assert "$defs" not in cleaned
    assert "title" not in cleaned
    assert cleaned["type"] == "object"
    assert cleaned["properties"]["item"]["properties"]["name"]["type"] == "string"


def test_openai_message_translation():
    msgs = [
        Message(role="user", text="hello"),
        Message(
            role="assistant",
            text="thinking",
            tool_calls=(ToolCall("call_123", "calculate", {"expression": "2+2"}),),
        ),
        Message(role="tool", tool_results=(ToolResult("call_123", "4", is_error=False),)),
    ]
    formatted = to_openai_messages("You are a helper", msgs)
    assert len(formatted) == 4
    assert formatted[0] == {"role": "system", "content": "You are a helper"}
    assert formatted[1] == {"role": "user", "content": "hello"}
    assert formatted[2]["role"] == "assistant"
    assert formatted[2]["content"] == "thinking"
    assert formatted[2]["tool_calls"][0]["id"] == "call_123"
    assert formatted[3]["role"] == "tool"
    assert formatted[3]["tool_call_id"] == "call_123"
    assert formatted[3]["content"] == "4"


def test_parse_openai_response():
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello world",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_999",
                            function=SimpleNamespace(
                                name="calculate",
                                arguments='{"expression": "5 * 5"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5),
        ),
    )
    r = parse_openai_response(resp, 45.0)
    assert r.text == "Hello world"
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].id == "call_999"
    assert r.tool_calls[0].name == "calculate"
    assert r.tool_calls[0].arguments == {"expression": "5 * 5"}
    assert r.usage.input_tokens == 20
    assert r.usage.output_tokens == 10
    assert r.usage.cache_read_tokens == 5
    assert r.stop_reason == "tool_calls"


def test_openai_complete_mock():
    class MockChatCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Mocked answer", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    prompt_tokens_details=None,
                ),
            )

    class MockAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=MockChatCompletions())

    client = OpenAIClient(model="gpt-4o-mini", api_key="sk-test", client=MockAsyncOpenAI())
    res = asyncio.run(
        client.complete(
            system="system prompt",
            messages=[Message(role="user", text="test")],
            tools=[],
            temperature=0.0,
            max_tokens=100,
        )
    )
    assert res.text == "Mocked answer"
    assert res.usage.input_tokens == 10
