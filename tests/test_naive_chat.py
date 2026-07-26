from __future__ import annotations

import json

from helicopter_cli.naive_chat import serialize_messages, serialize_openai_request


def test_serialize_messages_preserves_roles_order_and_content() -> None:
    messages = [
        {"role": "system", "content": "Keep this system message unchanged."},
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": "I will inspect the tool result.",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "bash"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "bash", "content": "output"},
    ]

    rendered = serialize_messages(messages)

    assert rendered.index("System: Keep this system message unchanged.") < rendered.index("User: Question")
    assert rendered.index("User: Question") < rendered.index("Assistant: I will inspect")
    assert "Assistant tool calls:" in rendered
    assert 'Tool call id: "call-1"' in rendered
    assert "Tool name: \"bash\"" in rendered
    assert "Tool: output" in rendered


def test_serialize_openai_request_moves_tool_metadata_into_transcript() -> None:
    request = {
        "model": "rwkv",
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "temperature": 0,
    }

    forwarded = serialize_openai_request(request)
    transcript = forwarded["messages"][0]["content"]

    assert forwarded["model"] == "rwkv"
    assert forwarded["temperature"] == 0
    assert forwarded.keys() >= {"model", "messages", "temperature"}
    assert not ({"tools", "tool_choice", "parallel_tool_calls"} & forwarded.keys())
    assert transcript.endswith("Assistant:")
    assert "System: S" in transcript
    assert "User: U" in transcript
    assert "OpenAI tools:" in transcript
    assert json.dumps(request["tools"], ensure_ascii=False, sort_keys=True, separators=(",", ":")) in transcript


def test_serialize_openai_request_rejects_missing_messages() -> None:
    try:
        serialize_openai_request({"model": "rwkv"})
    except ValueError as error:
        assert "messages array" in str(error)
    else:
        raise AssertionError("missing messages must be an explicit transport failure")
