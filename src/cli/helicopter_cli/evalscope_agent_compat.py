"""Strict wire compatibility for RWKV text tool calls and EvalScope FC.

RWKV naive Chat can return a valid BFCL-style JSON function map in
``message.content``.  EvalScope's OpenAI FC adapters consume
``message.tool_calls`` instead.  This module converts only schema-validated
JSON; it never invents a function name, argument, or answer.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Mapping


def _compatible_tool_name(name: str, tools: list[Any]) -> str:
    """Resolve only an exact or unique punctuation-equivalent tool name."""

    schemas = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        source = function if isinstance(function, Mapping) else tool
        candidate = source.get("name") if isinstance(source, Mapping) else None
        if isinstance(candidate, str) and candidate.strip():
            schemas.append(candidate.strip())
    if name in schemas:
        return name
    equivalents = [
        candidate
        for candidate in schemas
        if candidate.replace(".", "_") == name or candidate.replace("_", ".") == name
    ]
    if len(equivalents) == 1:
        return equivalents[0]
    raise ValueError(f"function name {name!r} is not an exact or unique compatible tool name")


def adapt_tool_call_response(
    response: Any,
    *,
    tools: list[Any] | None,
) -> tuple[Any, dict[str, Any]]:
    """Convert strict BFCL text calls in a completion response to tool_calls.

    The returned trace is diagnostic metadata.  The original response is
    still retained by the caller's upstream trace before the adapted response
    is handed to EvalScope.
    """

    trace: dict[str, Any] = {"status": "unchanged", "reason": None}
    if not isinstance(response, dict):
        trace["reason"] = "completion response is not an object"
        return response, trace
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        trace["reason"] = "completion response has no choices"
        return response, trace
    if not isinstance(tools, list) or not tools:
        trace["reason"] = "request contains no tool schemas"
        return response, trace

    # Import lazily to keep the compatibility module independent of the
    # router's import order; parse_candidates owns the strict schema checks.
    from .parallel_candidate_proxy import parse_candidates

    output = dict(response)
    output_choices = list(choices)
    converted = 0
    for index, original_choice in enumerate(choices):
        if not isinstance(original_choice, Mapping):
            continue
        original_message = original_choice.get("message")
        if not isinstance(original_message, Mapping):
            continue
        existing_calls = original_message.get("tool_calls")
        if isinstance(existing_calls, list) and existing_calls:
            trace["reason"] = "native tool_calls already present"
            continue
        content = original_message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            candidates = parse_candidates(content, tools=tools)
            native_calls = []
            for position, candidate in enumerate(candidates):
                native_name = _compatible_tool_name(candidate.name, tools)
                native_calls.append(
                    {
                        "id": f"call_compat_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": native_name,
                            "arguments": json.dumps(
                                candidate.arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
        except ValueError as error:
            trace["reason"] = "content was not a strictly valid schema-checked tool call"
            trace["error"] = str(error)
            continue
        if not native_calls:
            continue
        adapted_choice = dict(original_choice)
        adapted_message = dict(original_message)
        adapted_message["content"] = None
        adapted_message["tool_calls"] = native_calls
        adapted_choice["message"] = adapted_message
        adapted_choice["finish_reason"] = "tool_calls"
        output_choices[index] = adapted_choice
        converted += 1

    if converted:
        output["choices"] = output_choices
        trace["status"] = "converted"
        trace["choices_converted"] = converted
    return output, trace


__all__ = ["adapt_tool_call_response"]
