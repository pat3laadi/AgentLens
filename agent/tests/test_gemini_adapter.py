import asyncio
from types import SimpleNamespace

from agentlens_agent.llm.base import Message, ToolCall, ToolResult, ToolSpec
from agentlens_agent.llm.gemini_client import (
    GeminiClient,
    clean_schema_for_gemini,
    parse_gemini_response,
    to_gemini_contents,
)


def test_schema_cleaning():
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
            "optional_field": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }
    cleaned = clean_schema_for_gemini(raw_schema)
    assert "$defs" not in cleaned
    assert "title" not in cleaned
    assert cleaned["properties"]["item"]["type"] == "object"
    assert cleaned["properties"]["item"]["properties"]["name"]["type"] == "string"
    assert cleaned["properties"]["optional_field"] == {"type": "string", "nullable": True}


def test_message_translation():
    msgs = [
        Message(role="user", text="hello"),
        Message(
            role="assistant",
            text="thinking",
            tool_calls=(ToolCall("calculate:1234", "calculate", {"expression": "2+2"}),),
        ),
        Message(role="tool", tool_results=(ToolResult("calculate:1234", "4", is_error=False),)),
    ]
    contents = to_gemini_contents(msgs)
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert contents[2].role == "user"


def test_parse_gemini_response():
    resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="Here is the result"),
                        SimpleNamespace(
                            text=None,
                            function_call=SimpleNamespace(
                                name="calculate",
                                args={"expression": "10 * 5"},
                            ),
                        ),
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=15,
            candidates_token_count=8,
            cached_content_token_count=0,
        ),
    )
    r = parse_gemini_response(resp, 25.0)
    assert r.text == "Here is the result"
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "calculate"
    assert r.tool_calls[0].arguments == {"expression": "10 * 5"}
    assert r.usage.input_tokens == 15
    assert r.usage.output_tokens == 8


def test_complete_mock():
    class MockModels:
        async def generate_content(self, *, model, contents, config):
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[SimpleNamespace(text="Done!")]),
                        finish_reason="STOP",
                    )
                ],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=5,
                    candidates_token_count=2,
                    cached_content_token_count=0,
                ),
            )

    mock_client = SimpleNamespace(aio=SimpleNamespace(models=MockModels()))
    client = GeminiClient("gemini-2.5-flash", client=mock_client)
    res = asyncio.run(
        client.complete(
            system="system prompt",
            messages=[Message(role="user", text="test")],
            tools=[ToolSpec("test", "test desc", {"type": "object"})],
            temperature=0.0,
            max_tokens=100,
        )
    )
    assert res.text == "Done!"
    assert res.usage.input_tokens == 5
