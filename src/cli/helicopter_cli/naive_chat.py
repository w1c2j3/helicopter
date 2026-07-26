"""Naive Chat wire-format conversion for local RWKV OpenAI endpoints.

The evaluator owns the conversation semantics.  This module only converts the
already-built OpenAI chat request at the transport boundary into the plain
``System:``/``User:``/``Assistant:`` transcript expected by the local RWKV
checkpoint.  It never edits message content, invents an answer, or parses a
model response.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


ROLE_LABELS = {
    "system": "System",
    "developer": "Developer",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
}
_TOOL_REQUEST_KEYS = ("tools", "tool_choice", "parallel_tool_calls")


def _json(value: Any) -> str:
    """Render request metadata without losing fields or relying on repr()."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _json(value)


def _message_lines(message: Mapping[str, Any]) -> list[str]:
    role = str(message.get("role", "user"))
    label = ROLE_LABELS.get(role, role.title())
    lines = [f"{label}: {_content(message.get('content', ''))}"]
    if role == "assistant" and message.get("tool_calls") is not None:
        lines.append(f"Assistant tool calls: {_json(message['tool_calls'])}")
    if role == "tool":
        if message.get("tool_call_id") is not None:
            lines.append(f"Tool call id: {_json(message['tool_call_id'])}")
        if message.get("name") is not None:
            lines.append(f"Tool name: {_json(message['name'])}")
    return lines


def serialize_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Serialize all existing chat messages in their original order.

    Separating messages by blank lines makes the boundaries unambiguous while
    keeping every original content value untouched.  Unknown roles are kept
    instead of being silently dropped.
    """

    return "\n\n".join("\n".join(_message_lines(message)) for message in messages)


def serialize_openai_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return the transport-only naive Chat representation of an OpenAI call.

    Tool schemas and selection metadata are included as JSON records because
    removing them would change the request semantics.  The upstream endpoint
    does not receive the unsupported OpenAI tool fields; it receives their
    textual representation as part of the same serialized conversation.
    """

    messages = request.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("OpenAI request must contain a messages array")

    sections = [serialize_messages([message]) for message in messages]
    for key in _TOOL_REQUEST_KEYS:
        if key in request:
            sections.append(f"OpenAI {key}: {_json(request[key])}")
    transcript = "\n\n".join(sections)
    if transcript:
        transcript += "\n\n"
    transcript += "Assistant:"

    forwarded = {key: value for key, value in request.items() if key not in _TOOL_REQUEST_KEYS}
    forwarded["messages"] = [{"role": "user", "content": transcript}]
    return forwarded


__all__ = ["serialize_messages", "serialize_openai_request"]
