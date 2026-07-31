from __future__ import annotations

import json

from helicopter_cli.evalscope_agent_compat import adapt_tool_call_response


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


def test_adapt_bfcl_content_to_native_tool_calls() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<think></think>\n```json\n[{"bash":{"command":"echo hi"}}]\n```',
                },
                "finish_reason": "stop",
            }
        ]
    }

    adapted, trace = adapt_tool_call_response(response, tools=TOOLS)

    message = adapted["choices"][0]["message"]
    assert trace == {"status": "converted", "reason": None, "choices_converted": 1}
    assert message["content"] is None
    assert adapted["choices"][0]["finish_reason"] == "tool_calls"
    assert message["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"command": "echo hi"}


def test_adapt_accepts_unique_dot_underscore_tool_name_alias() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "uber.ride",
                "parameters": {
                    "type": "object",
                    "properties": {"type": {"type": "string"}},
                    "required": ["type"],
                },
            },
        }
    ]
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": '[{"uber_ride":{"type":"comfort"}}]'},
                "finish_reason": "stop",
            }
        ]
    }

    adapted, trace = adapt_tool_call_response(response, tools=tools)

    assert trace["status"] == "converted"
    assert adapted["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "uber.ride"


def test_adapt_keeps_invalid_content_and_records_reason() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": '[{"bash":{"unknown":1}}]'},
                "finish_reason": "stop",
            }
        ]
    }

    adapted, trace = adapt_tool_call_response(response, tools=TOOLS)

    assert adapted == response
    assert trace["status"] == "unchanged"
    assert "unknown fields" in trace["error"]


def test_adapt_preserves_existing_native_tool_calls() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-native",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    adapted, trace = adapt_tool_call_response(response, tools=TOOLS)

    assert adapted == response
    assert trace["reason"] == "native tool_calls already present"
