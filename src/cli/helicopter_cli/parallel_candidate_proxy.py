"""Local Agent proxy for RWKV endpoints without native OpenAI tool parsing.

The proxy is deliberately an evaluation-layer adapter.  It does not alter the
EvalScope task, system messages, or tool semantics.  It embeds the original
conversation verbatim in short routing prompts, asks the upstream model for
strict JSON candidates without sending OpenAI ``tools`` fields, and exposes a
candidate as a normal OpenAI ``message.tool_calls`` response.

Every source request, candidate request/response, aggregate request/response,
selection decision, and final response is written to JSONL.  Strictly valid
candidates are preferred for routing; a syntactically recognizable call that
fails the tool schema is still transported as a native tool call so the
benchmark can score the invalid arguments instead of receiving a text
fallback.  The transport layer does not decide whether a name or argument is
correct; it only preserves the model's call shape for the evaluator.
"""

from __future__ import annotations

import base64
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .evalscope_agent_compat import adapt_tool_call_response
from .rwkv_agent_prompt import (
    DEFAULT_LONG_DOC_MAX_CHARS,
    DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS,
    DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS,
    DEFAULT_LONG_DOC_MIN_CHARS,
    DEFAULT_LONG_DOC_OVERLAP_LINES,
    LongContextConfig,
    build_rwkv_json_call_prompt,
    compact_messages_for_long_context,
    normalize_messages,
    normalize_rwkv_text,
)


@dataclass(frozen=True, slots=True)
class ParallelCandidateConfig:
    """Fixed, reproducible limits for the local candidate route."""

    chunk_tools: int = 2
    batch_size: int = 16
    context_chars: int = 6000
    prompt_max_chars: int = 12288
    candidate_max_tokens: int = 2048
    aggregate_max_tokens: int = 2048
    max_candidates: int = 12
    fallback_to_highest_confidence: bool = True
    fallback_to_native_chat: bool = True
    long_doc_min_chars: int = DEFAULT_LONG_DOC_MIN_CHARS
    long_doc_max_chars: int = DEFAULT_LONG_DOC_MAX_CHARS
    long_doc_overlap_lines: int = DEFAULT_LONG_DOC_OVERLAP_LINES
    long_doc_max_evidence_chunks: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS
    long_doc_max_evidence_chars: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    arguments: dict[str, Any]
    confidence: float
    evidence: str
    arguments_text: str | None = None


def _candidate_arguments_text(candidate: Candidate) -> str:
    """Serialize arguments without changing an invalid raw payload."""

    if candidate.arguments_text is not None:
        return candidate.arguments_text
    return json.dumps(candidate.arguments, ensure_ascii=False, separators=(",", ":"))


# Match the official RWKV NoCoT function-calling prefill used by rwkv-skills.
# The empty think block is part of request-time serialization only; source
# messages and their semantics remain unchanged.
_NO_COT_JSON_ASSISTANT_PREFIX = "Assistant: <think></think>\n```json\n"
_FLOWER_JSON_STOP_SUFFIXES = [
    "\n```",
    "```",
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
    "\nUser\u273f",
    "User\u273f",
    "\nBot\u273f",
    "Bot\u273f",
    "\u273f",
]


def _redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: ("Bearer [redacted]" if key.lower() == "authorization" else value)
        for key, value in headers.items()
    }


def _json_or_text(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body.decode("utf-8", "replace")


def _usage_from_upstream(trace: Any) -> dict[str, int] | None:
    """Read OpenAI usage metadata from one raw completion trace."""

    if not isinstance(trace, Mapping):
        return None
    response = trace.get("response")
    if not isinstance(response, Mapping):
        return None
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    values: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values[key] = int(value)
    return values


def _sum_upstream_usage(traces: list[Any]) -> dict[str, int] | None:
    """Aggregate usage for all hidden candidate and aggregate requests."""

    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    found = False
    for trace in traces:
        usage = _usage_from_upstream(trace)
        if usage is None:
            continue
        found = True
        for key in total:
            total[key] += usage[key]
    return total if found else None


def _response_usage(usage: dict[str, int] | None) -> dict[str, int]:
    """Return OpenAI usage metadata required by strict FC consumers.

    Some upstream responses, especially empty non-tool completions, omit
    ``usage`` entirely. The response adapter must still return a valid
    OpenAI envelope because BFCL reads the field before it can score the
    model's actual message/tool-call content. Zeroes are telemetry-only;
    they never alter content, tool calls, arguments, or evaluator results.
    """

    return usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _json_default(value: Any) -> Any:
    """Keep non-JSON upstream trace values losslessly serializable."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "__type__": "bytes",
            "base64": base64.b64encode(bytes(value)).decode("ascii"),
        }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _function_schema(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, Mapping):
        return None
    function = tool.get("function")
    if isinstance(function, Mapping):
        name = str(function.get("name") or "").strip()
        if not name:
            return None
        return {
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": function.get("parameters") if isinstance(function.get("parameters"), Mapping) else {},
        }
    name = str(tool.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "description": str(tool.get("description") or ""),
        "parameters": tool.get("parameters") if isinstance(tool.get("parameters"), Mapping) else {},
    }


def _schemas(tools: list[Any]) -> list[dict[str, Any]]:
    return [schema for tool in tools if (schema := _function_schema(tool)) is not None]


def _schema_by_name(tools: list[Any]) -> dict[str, dict[str, Any]]:
    return {schema["name"]: schema for schema in _schemas(tools)}


def _resolve_schema_name(name: str, schemas: Mapping[str, Any]) -> str:
    """Resolve exact names and only unique dot/underscore wire aliases."""

    if name in schemas:
        return name
    equivalents = [
        candidate
        for candidate in schemas
        if candidate.replace(".", "_") == name or candidate.replace("_", ".") == name
    ]
    if len(equivalents) == 1:
        return equivalents[0]
    if not equivalents:
        raise ValueError(f"candidate name {name!r} is not in the supplied tools")
    raise ValueError(f"candidate name {name!r} is not an exact or unique compatible tool name")


def _json_schema_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _validate_json_schema(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    """Validate generated arguments without coercing or filling values."""

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise ValueError(f"{path} must be one of {enum!r}")

    alternatives = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(alternatives, list) and alternatives:
        errors: list[str] = []
        for alternative in alternatives:
            if not isinstance(alternative, Mapping):
                continue
            try:
                _validate_json_schema(value, alternative, path=path)
            except ValueError as error:
                errors.append(str(error))
            else:
                break
        else:
            raise ValueError(f"{path} does not match any allowed schema: {errors[-1] if errors else 'no schema'}")
        return

    expected = schema.get("type")
    expected_types = [expected] if isinstance(expected, str) else expected if isinstance(expected, list) else []
    if expected_types:
        compatible = any(
            _json_schema_type(value) == item
            or (item == "number" and _json_schema_type(value) == "integer")
            for item in expected_types
        )
        if not compatible:
            label = expected_types[0] if len(expected_types) == 1 else expected_types
            raise ValueError(f"{path} must be {label}, got {_json_schema_type(value)}")

    if isinstance(value, Mapping):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        additional = schema.get("additionalProperties", False if properties else True)
        if additional is False:
            unknown = set(value).difference(str(key) for key in properties)
            if unknown:
                if path == "candidate arguments":
                    raise ValueError(f"candidate arguments contain unknown fields: {sorted(unknown)}")
                raise ValueError(f"{path} contains unknown fields: {sorted(unknown)}")
        required = schema.get("required")
        if isinstance(required, list):
            missing = [str(key) for key in required if key not in value]
            if missing:
                if path == "candidate arguments":
                    raise ValueError(f"candidate is missing required arguments: {missing}")
                raise ValueError(f"{path} is missing required fields: {missing}")
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, Mapping):
                _validate_json_schema(value[key], child_schema, path=f"{path}.{key}")
    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _validate_json_schema(item, items, path=f"{path}[{index}]")


def _message_content(message: Any) -> str:
    if not isinstance(message, Mapping):
        return str(message or "")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _json_value(text: str) -> dict[str, Any] | list[Any]:
    """Extract one complete model-generated candidate JSON value.

    RWKV may emit a reasoning segment followed by an explicit ``</think>``
    delimiter and the requested JSON value.  The delimiter is part of the
    model output; using only the suffix after that delimiter keeps extraction
    deterministic while preserving strict schema validation below.  A
    single-element array is retained as an explicit transport normalization
    because the aggregate model has been observed to wrap one candidate in
    ``[{...}]``.  The caller still rejects empty/multi-element arrays and
    validates the unwrapped object without filling or repairing fields.
    """

    source = str(text or "").strip()
    closing_think = source.rfind("</think>")
    if closing_think >= 0:
        suffix = source[closing_think + len("</think>") :].strip()
        if suffix:
            source = suffix
    if source.startswith("```"):
        lines = source.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        source = "\n".join(lines).strip()
    if source.startswith("Assistant:"):
        source = source[len("Assistant:") :].lstrip()
        if source.startswith("```"):
            lines = source.splitlines()
            lines = lines[1:] if lines else lines
            source = "\n".join(lines).strip()
    if not source.startswith(("{", "[")):
        raise ValueError("completion must start with a JSON object or array")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(source)
    except json.JSONDecodeError as error:
        raise ValueError("completion did not contain a complete JSON object") from error
    trailing = source[end:].strip()
    if trailing.startswith("```"):
        trailing = trailing[3:].strip()
    if trailing:
        raise ValueError("completion contains text after the JSON object")
    if not isinstance(value, (dict, list)):
        raise ValueError("completion JSON value must be an object or array")
    return value


def _parse_native_candidate_envelope(value: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Parse model-emitted OpenAI-style tool-call envelopes.

    This is transport normalization only.  Worker prompts still produce one
    candidate, while the aggregate prompt may return several independent
    calls for a multi-action user request.  Every returned call is validated
    against the supplied tool schema by the caller; no call is selected or
    repaired here.
    """

    extra_envelope_keys = sorted(str(key) for key in value if str(key) != "tool_calls")
    if extra_envelope_keys:
        raise ValueError(f"candidate native envelope has unsupported fields: {extra_envelope_keys}")
    calls = value.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("candidate native tool_calls must contain at least one call")
    parsed: list[tuple[str, dict[str, Any]]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise ValueError("candidate native tool call must be an object")
        extra_call_keys = sorted(str(key) for key in call if str(key) not in {"id", "type", "index", "function"})
        if extra_call_keys:
            raise ValueError(f"candidate native tool call has unsupported fields: {extra_call_keys}")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("candidate native tool call must contain function")
        extra_function_keys = sorted(str(key) for key in function if str(key) not in {"name", "arguments"})
        if extra_function_keys:
            raise ValueError(f"candidate native function has unsupported fields: {extra_function_keys}")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("candidate native function name must be a non-empty string")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ValueError("candidate native function arguments must contain a JSON object") from error
        if not isinstance(arguments, Mapping):
            raise ValueError("candidate native function arguments must be a JSON object")
        parsed.append((name.strip(), dict(arguments)))
    return parsed


def _compact_prompt_parameter_schema(value: Any) -> dict[str, Any]:
    """Keep schema facts needed by the worker while bounding prompt size."""

    if not isinstance(value, Mapping):
        return {"type": "string"}
    compact: dict[str, Any] = {"type": str(value.get("type") or "string")}
    description = normalize_rwkv_text(str(value.get("description") or ""))
    if description:
        compact["description"] = description[:48]
    enum = value.get("enum")
    if isinstance(enum, list) and len(enum) <= 12:
        compact["enum"] = list(enum)
    items = value.get("items")
    if isinstance(items, Mapping):
        compact["items"] = _compact_prompt_parameter_schema(items)
    properties = value.get("properties")
    if isinstance(properties, Mapping):
        compact["properties"] = {
            str(name): _compact_prompt_parameter_schema(schema)
            for name, schema in properties.items()
        }
    required = value.get("required")
    if isinstance(required, list):
        compact["required"] = [str(name) for name in required]
    return compact


def _compact_prompt_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    return {
        "name": str(schema.get("name") or ""),
        "description": normalize_rwkv_text(str(schema.get("description") or ""))[:120],
        "parameters": _compact_prompt_parameter_schema(parameters),
    }


def _candidate_values(value: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else [value]
    output: list[dict[str, Any]] = []
    candidate_fields = {"name", "arguments", "confidence", "evidence", "id", "tool_call_id", "tool_calls"}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("completion JSON array candidate must be an object")
        if "tool_calls" in item:
            output.extend(
                {"name": name, "arguments": arguments}
                for name, arguments in _parse_native_candidate_envelope(item)
            )
        elif len(item) == 1 and next(iter(item)) not in candidate_fields:
            # BFCL's text form is an array of one-entry function maps, while
            # the OpenAI wire form is message.tool_calls.  Keep the conversion
            # lossless: the schema validator below still rejects unknown
            # functions, malformed arguments, and missing required fields.
            name, arguments = next(iter(item.items()))
            output.append({"name": name, "arguments": arguments})
        else:
            output.append(item)
    return output


def _parse_candidate_value(value: dict[str, Any], *, tools: list[Any]) -> Candidate:
    unknown = set(value).difference({"name", "arguments", "confidence", "evidence", "id", "tool_call_id"})
    if unknown:
        raise ValueError(f"candidate contains unknown fields: {sorted(unknown)}")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("candidate name must be a non-empty string")
    name = name.strip()
    schemas = _schema_by_name(tools)
    name = _resolve_schema_name(name, schemas)
    arguments = value.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError("candidate arguments string must contain a JSON object") from error
    if not isinstance(arguments, dict):
        raise ValueError("candidate arguments must be a JSON object")
    parameters = schemas[name].get("parameters")
    if isinstance(parameters, Mapping):
        _validate_json_schema(arguments, parameters, path="candidate arguments")
    confidence = value.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("candidate confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("candidate confidence must be between 0 and 1")
    evidence = value.get("evidence", "")
    if not isinstance(evidence, str):
        raise ValueError("candidate evidence must be a string")
    return Candidate(name=name, arguments=dict(arguments), confidence=float(confidence), evidence=evidence)


def parse_candidates(text: str, *, tools: list[Any]) -> list[Candidate]:
    """Strictly validate one or more model-generated tool-call candidates."""

    value = _json_value(text)
    values = _candidate_values(value)
    if not values:
        raise ValueError("completion JSON array must contain at least one candidate")
    return [_parse_candidate_value(item, tools=tools) for item in values]


def parse_candidate(text: str, *, tools: list[Any]) -> Candidate:
    """Strictly validate exactly one candidate against the supplied schemas."""

    value = _json_value(text)
    if isinstance(value, list) and len(value) != 1:
        raise ValueError("completion JSON array must contain exactly one candidate")
    candidates = parse_candidates(text, tools=tools)
    if len(candidates) != 1:
        raise ValueError("candidate native tool_calls must contain exactly one call")
    return candidates[0]


def _transport_candidate_values(value: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """Normalize candidate wire shapes without applying the tool schema.

    The benchmark must see a structured tool call even when its arguments are
    wrong.  Schema validation is useful for candidate selection, but applying
    it to the transport boundary turns ordinary model mistakes (for example
    ``null`` for an optional numeric argument) into a text response that the
    BFCL adapter cannot score.  This helper only performs shape normalization;
    the caller still decides how the call should be scored.
    """

    values = value if isinstance(value, list) else [value]
    output: list[dict[str, Any]] = []
    candidate_fields = {"name", "arguments", "confidence", "evidence", "id", "tool_call_id", "tool_calls"}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("completion JSON array candidate must be an object")
        if "tool_calls" in item:
            calls = item.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError("candidate native tool_calls must contain at least one call")
            for call in calls:
                if not isinstance(call, Mapping):
                    raise ValueError("candidate native tool call must be an object")
                function = call.get("function")
                if not isinstance(function, Mapping):
                    raise ValueError("candidate native tool call must contain function")
                name = function.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("candidate native function name must be a non-empty string")
                output.append({"name": name.strip(), "arguments": function.get("arguments", {})})
        elif len(item) == 1 and next(iter(item)) not in candidate_fields:
            name, arguments = next(iter(item.items()))
            output.append({"name": name, "arguments": arguments})
        else:
            output.append(item)
    return output


def _parse_transport_candidate_value(value: dict[str, Any], *, tools: list[Any]) -> Candidate:
    """Parse one known-tool call while preserving schema-invalid arguments."""

    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("candidate name must be a non-empty string")
    # Transport normalization deliberately does not resolve the name against
    # the supplied schema.  Whether a tool exists, and whether its arguments
    # are correct, belongs to EvalScope/BFCL rather than this adapter.
    name = name.strip()
    arguments = value.get("arguments", {})
    arguments_text: str | None = None
    if isinstance(arguments, str):
        raw_arguments = arguments
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            # Preserve the raw transport payload.  EvalScope can expose its
            # parse error to the benchmark rather than this adapter deciding
            # what the malformed payload means.
            arguments_text = raw_arguments
            arguments = {}
        else:
            if isinstance(parsed_arguments, dict):
                arguments = parsed_arguments
            else:
                # A JSON scalar/array is still a valid transport payload, but
                # it is not a function-arguments object.  Keep the exact text
                # for the native tool-call envelope; schema/correctness
                # handling remains EvalScope/BFCL's responsibility.
                arguments_text = raw_arguments
                arguments = {}
    elif not isinstance(arguments, dict):
        arguments_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        arguments = {}
    confidence = value.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))
    evidence = value.get("evidence", "")
    if not isinstance(evidence, str):
        evidence = ""
    return Candidate(
        name=name,
        arguments=dict(arguments),
        confidence=confidence,
        evidence=evidence,
        arguments_text=arguments_text,
    )


def parse_transport_candidates(text: str, *, tools: list[Any]) -> list[Candidate]:
    """Normalize known-tool calls without schema validation for transport."""

    try:
        value = _json_value(text)
    except ValueError as error:
        # Some BFCL parallel/multiple outputs are emitted as adjacent JSON
        # objects (``{...}{...}``) rather than a JSON array.  Split only
        # complete top-level JSON values; this changes transport shape and
        # leaves all correctness/schema decisions to EvalScope/BFCL.
        source = str(text or "").strip()
        decoder = json.JSONDecoder()
        values: list[Any] = []
        while source:
            try:
                item, end = decoder.raw_decode(source)
            except json.JSONDecodeError:
                break
            values.append(item)
            source = source[end:].lstrip()
            if source.startswith(","):
                source = source[1:].lstrip()
        if len(values) < 2 or source or not all(isinstance(item, dict) for item in values):
            raise error
        value = values
    values = _transport_candidate_values(value)
    if not values:
        raise ValueError("completion JSON array must contain at least one candidate")
    return [_parse_transport_candidate_value(item, tools=tools) for item in values]


def parse_transport_candidate(text: str, *, tools: list[Any]) -> Candidate:
    """Normalize exactly one known-tool call without schema validation."""

    candidates = parse_transport_candidates(text, tools=tools)
    if len(candidates) != 1:
        raise ValueError("candidate transport payload must contain exactly one call")
    return candidates[0]


def _compact_prompt_tools(tools: list[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for schema in _schemas(tools):
        output.append(_compact_prompt_tool_schema(schema))
    return output


def _compact_prompt_tool_names(tools: list[Any]) -> list[str]:
    """Return the validated tool-name catalog used by the aggregator.

    Candidate arguments have already been schema-validated before the
    aggregator runs. Repeating every parameter schema in the aggregate prompt
    makes large Agent tool inventories exceed the model context budget without
    adding information needed to choose among the candidates.
    """

    return [str(schema["name"]) for schema in _schemas(tools)]


def _compact_aggregate_argument_catalog(
    candidates: list[Candidate],
    tools: list[Any],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    """Bound the aggregate prompt's argument-key catalog.

    The aggregate model may need to add an action that was not emitted by a
    worker candidate.  Tool names alone are insufficient for that case: the
    model can copy a valid-looking key from one tool to another.  Keep the
    candidate tools first, then add the remaining tools until the explicit
    prompt budget is reached.  This catalog is advisory only; every emitted
    action is still validated by :func:`parse_candidates`.
    """

    schemas = _schema_by_name(tools)
    ordered_names: list[str] = []
    for name in [candidate.name for candidate in candidates] + list(schemas):
        if name not in ordered_names:
            ordered_names.append(name)
    rows: list[dict[str, Any]] = []
    used = 0
    for name in ordered_names:
        schema = schemas[name]
        parameters = schema.get("parameters") if isinstance(schema.get("parameters"), Mapping) else {}
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), Mapping) else {}
        row: dict[str, Any] = {
            "name": name,
            "allowed_argument_names": sorted(str(key) for key in properties),
        }
        required = parameters.get("required")
        if isinstance(required, list):
            row["required_argument_names"] = [str(key) for key in required]
        if parameters.get("additionalProperties") is False:
            row["additional_properties"] = False
        encoded_length = len(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        if rows and used + encoded_length > max(1, int(max_chars)):
            break
        rows.append(row)
        used += encoded_length
    return rows


def _candidate_prompt(
    tools: list[Any],
    messages: list[dict[str, str]],
    *,
    config: ParallelCandidateConfig,
) -> tuple[str, dict[str, Any]]:
    system_prompt = "\n".join(
        [
            "You are a worker in a parallel candidate tool-call router.",
            "Choose the single best next tool action for the original conversation.",
            "Return exactly one JSON object with only these fields:",
            '{"name":"tool_name","arguments":{},"confidence":0.0,"evidence":"short reason"}',
            "Use only a name from this shard. Do not invent identifiers or arguments.",
            "Do not include id, type, tool_calls, function, analysis, markdown, or extra fields.",
            "Tools:",
            json.dumps(_compact_prompt_tools(tools), ensure_ascii=False, separators=(",", ":")),
        ]
    )
    return build_rwkv_json_call_prompt(
        system_prompt,
        messages,
        history_max_chars=config.context_chars,
        prompt_max_chars=config.prompt_max_chars,
        assistant_prefix=_NO_COT_JSON_ASSISTANT_PREFIX,
        prompt_style="assistant",
    )


def _aggregate_prompt(
    candidates: list[Candidate],
    tools: list[Any],
    messages: list[dict[str, str]],
    *,
    config: ParallelCandidateConfig,
) -> tuple[str, dict[str, Any]]:
    ranked_candidates = sorted(candidates, key=lambda item: item.confidence, reverse=True)[
        : max(1, int(config.max_candidates))
    ]
    rows = [
        {
            "name": item.name,
            "arguments": item.arguments,
            "confidence": item.confidence,
            "evidence": item.evidence,
        }
        for item in ranked_candidates
    ]
    system_prompt = "\n".join(
        [
            "You are the aggregator for a parallel candidate tool-call router.",
            "Choose the best next action from the candidates for the original conversation.",
            "Return exactly one JSON array of one or more objects with only these fields:",
            '[{"name":"tool_name","arguments":{},"confidence":0.0,"evidence":"short reason"}]',
            "Use only supplied tool names. Preserve each candidate argument object exactly when selecting it; add another array item only for another independent action explicitly requested by the conversation.",
            "For every emitted item, use only the argument keys allowed for that tool. Never rename, borrow, or invent argument keys.",
            "Do not include id, type, tool_calls, function, analysis, markdown, or extra fields.",
            "Candidates:",
            json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
            "Valid tool names:",
            json.dumps(_compact_prompt_tool_names(tools), ensure_ascii=False, separators=(",", ":")),
            "Tool argument key catalog (bounded; final schema validation is authoritative):",
            json.dumps(
                _compact_aggregate_argument_catalog(
                    ranked_candidates,
                    tools,
                    max_chars=max(512, int(config.prompt_max_chars) // 3),
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
    )
    return build_rwkv_json_call_prompt(
        system_prompt,
        messages,
        history_max_chars=config.context_chars,
        prompt_max_chars=config.prompt_max_chars,
        assistant_prefix=_NO_COT_JSON_ASSISTANT_PREFIX,
        prompt_style="assistant",
    )


class ParallelCandidateProxy:
    """Process-local OpenAI-compatible Agent proxy."""

    def __init__(
        self,
        upstream_base_url: str,
        *,
        api_key: str,
        trace_path: Path,
        config: ParallelCandidateConfig | None = None,
    ) -> None:
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.api_key = api_key
        self.trace_path = trace_path
        self.config = config or ParallelCandidateConfig()
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("parallel-candidate proxy is not started")
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def start(self) -> str:
        if self._server is not None:
            return self.base_url
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _handle(self) -> None:
                started = perf_counter()
                request_headers = {key: value for key, value in self.headers.items()}
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                source_payload: Any = _json_or_text(body)
                response_status = 502
                response_headers: dict[str, str] = {"Content-Type": "application/json"}
                response_body = b""
                route_trace: dict[str, Any] = {"mode": "direct"}
                error: dict[str, str] | None = None
                try:
                    if self.command == "POST" and self.path.split("?", 1)[0].endswith("/chat/completions"):
                        if not isinstance(source_payload, dict):
                            raise ValueError("chat completion body must be a JSON object")
                        response_payload, route_trace = proxy._route(source_payload)
                        response_status = 200
                        response_body = json.dumps(response_payload, ensure_ascii=False).encode("utf-8")
                    else:
                        response_status, response_headers, response_body = proxy._forward(
                            self.path,
                            body,
                            method=self.command,
                        )
                except HTTPError as exc:
                    response_status = exc.code
                    response_headers = {key: value for key, value in exc.headers.items()}
                    response_body = exc.read()
                    error = {"type": "HTTPError", "message": str(exc)}
                except (OSError, URLError, ValueError) as exc:
                    response_status = 502
                    response_body = json.dumps(
                        {"error": {"type": type(exc).__name__, "message": str(exc)}},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    error = {"type": type(exc).__name__, "message": str(exc)}

                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "method": self.command,
                    "path": self.path,
                    "request": {"headers": _redact_headers(request_headers), "json": source_payload},
                    "response": {
                        "status": response_status,
                        "headers": response_headers,
                        "body": _json_or_text(response_body),
                    },
                    "router": route_trace,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "error": error,
                }
                with proxy._lock:
                    with proxy.trace_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

                self.send_response(response_status)
                for key, value in response_headers.items():
                    if key.lower() not in {"content-length", "transfer-encoding", "connection"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def do_GET(self) -> None:  # noqa: N802
                self._handle()

            def do_POST(self) -> None:  # noqa: N802
                self._handle()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="helicopter-parallel-candidate", daemon=True)
        self._thread.start()
        return self.base_url

    def _upstream_url(self, request_path: str) -> str:
        upstream = urlsplit(self.upstream_base_url)
        incoming = urlsplit(request_path)
        base_path = upstream.path.rstrip("/")
        suffix = incoming.path
        if base_path and suffix.startswith(base_path):
            suffix = suffix[len(base_path) :]
        url = f"{upstream.scheme}://{upstream.netloc}{base_path}{suffix}"
        return f"{url}?{incoming.query}" if incoming.query else url

    def _request_upstream(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], Any, dict[str, Any]]:
        started = perf_counter()
        outgoing_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            # vllm-rwkv's naive Chat contract is exposed through the raw
            # completions endpoint.  Sending the already-rendered transcript
            # to /chat/completions would apply the server chat template a
            # second time and produces the observed `></think>` completion.
            self._upstream_url("/v1/completions"),
            data=outgoing_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        status = 502
        headers: dict[str, str] = {}
        body: Any = None
        error: dict[str, str] | None = None
        try:
            with urlopen(request, timeout=180) as response:  # noqa: S310 - configured local endpoint
                status = response.status
                headers = dict(response.headers.items())
                body_bytes = response.read()
                body = _json_or_text(body_bytes)
        except HTTPError as exc:
            status = exc.code
            headers = {key: value for key, value in exc.headers.items()}
            body = _json_or_text(exc.read())
            error = {"type": "HTTPError", "message": str(exc)}
        except (OSError, URLError) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            body = {"error": error}
        return status, headers, body, {
            "url": self._upstream_url("/v1/completions"),
            "headers": {"Authorization": "Bearer [redacted]", "Content-Type": "application/json"},
            "json": payload,
            "status": status,
            "response": body,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "error": error,
        }

    def _forward(self, request_path: str, body: bytes, *, method: str) -> tuple[int, dict[str, str], bytes]:
        request = Request(
            self._upstream_url(request_path),
            data=body if body else None,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method=method,
        )
        with urlopen(request, timeout=60) as response:  # noqa: S310 - configured local endpoint
            return response.status, dict(response.headers.items()), response.read()

    def _native_chat_fallback(self, source: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Forward the unchanged source request when candidate routing fails.

        Candidate routing is an evaluation adapter, not a second tool protocol.
        If it cannot produce a validated candidate, preserve the upstream
        native response instead of converting prose into an empty tool call.
        The source payload is serialized unchanged so the upstream parser sees
        the original messages and tool definitions.
        """

        started = perf_counter()
        body: Any = None
        status = 502
        headers: dict[str, str] = {}
        error: dict[str, str] | None = None
        try:
            status, headers, body_bytes = self._forward(
                "/v1/chat/completions",
                json.dumps(source, ensure_ascii=False).encode("utf-8"),
                method="POST",
            )
            body = _json_or_text(body_bytes)
        except HTTPError as exc:
            status = exc.code
            headers = {key: value for key, value in exc.headers.items()}
            body = _json_or_text(exc.read())
            error = {"type": "HTTPError", "message": str(exc)}
        except (OSError, URLError) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            body = {"error": error}

        trace = {
            "url": self._upstream_url("/v1/chat/completions"),
            "headers": {"Authorization": "Bearer [redacted]", "Content-Type": "application/json"},
            "json": source,
            "status": status,
            "response": body,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "error": error,
        }
        if status < 400 and isinstance(body, dict):
            adapted, compatibility = adapt_tool_call_response(
                body,
                tools=source.get("tools") if isinstance(source.get("tools"), list) else None,
            )
            trace["compatibility"] = compatibility
            return adapted, trace
        trace["compatibility"] = {"status": "unchanged", "reason": "upstream response was not a successful JSON object"}
        return None, trace

    @staticmethod
    def _completion_text(response: Any) -> str:
        if not isinstance(response, Mapping):
            return ""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            return ""
        text = choices[0].get("text")
        if isinstance(text, str):
            return text
        message = choices[0].get("message")
        return _message_content(message)

    def _upstream_payload(self, source: dict[str, Any], prompt: str, *, max_tokens: int) -> dict[str, Any]:
        allowed = {
            "model",
            "temperature",
            "top_p",
            "seed",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "logprobs",
            "top_logprobs",
        }
        payload = {key: source[key] for key in allowed if key in source}
        payload["prompt"] = prompt
        payload["max_tokens"] = int(max_tokens)
        # The model-generated JSON must stop at the first response boundary.
        # These are transport stops, not answer repair: strict extraction below
        # still rejects malformed, incomplete, or schema-invalid JSON.
        payload.setdefault("stop", list(_FLOWER_JSON_STOP_SUFFIXES))
        payload["stream"] = False
        return payload

    def _route(self, source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        tools = source.get("tools")
        if not isinstance(tools, list) or not _schemas(tools):
            status, headers, body_bytes = self._forward(
                "/v1/chat/completions",
                json.dumps(source, ensure_ascii=False).encode("utf-8"),
                method="POST",
            )
            body = _json_or_text(body_bytes)
            return body if isinstance(body, dict) else {"choices": []}, {
                "mode": "direct",
                "upstream_status": status,
                "upstream_headers": headers,
                "upstream_response": body,
            }
        schemas = _schemas(tools)
        source_messages = source.get("messages")
        if not isinstance(source_messages, list) or not all(isinstance(item, Mapping) for item in source_messages):
            raise ValueError("messages must be an array of objects")
        normalized_messages = normalize_messages(source_messages)
        routed_messages, context_trace = compact_messages_for_long_context(
            normalized_messages,
            config=LongContextConfig(
                min_long_text_chars=self.config.long_doc_min_chars,
                max_chunk_chars=self.config.long_doc_max_chars,
                overlap_lines=self.config.long_doc_overlap_lines,
                max_evidence_chunks=self.config.long_doc_max_evidence_chunks,
                max_evidence_chars=self.config.long_doc_max_evidence_chars,
            ),
        )
        size = max(1, int(self.config.chunk_tools))
        shards = [schemas[index : index + size] for index in range(0, len(schemas), size)]
        candidate_traces: list[dict[str, Any]] = []
        valid_candidates: list[Candidate] = []
        transport_candidates: list[Candidate] = []

        def ask(shard: list[dict[str, Any]]) -> tuple[dict[str, Any], list[Candidate]]:
            prompt, prompt_trace = _candidate_prompt(shard, routed_messages, config=self.config)
            if prompt_trace["prompt_over_budget"]:
                return {
                    "tools": shard,
                    "prompt": prompt,
                    "completion": "",
                    "finish_reason": "prompt_over_budget",
                    "prompt_trace": prompt_trace,
                    "error": f"candidate prompt length {len(prompt)} exceeds {self.config.prompt_max_chars}",
                }, []
            payload = self._upstream_payload(source, prompt, max_tokens=self.config.candidate_max_tokens)
            status, _headers, body, raw = self._request_upstream(payload)
            completion = self._completion_text(body)
            trace: dict[str, Any] = {
                "tools": shard,
                "prompt": prompt,
                "completion": completion,
                "finish_reason": (
                    body.get("choices", [{}])[0].get("finish_reason", "stop")
                    if isinstance(body, Mapping) and isinstance(body.get("choices"), list) and body.get("choices")
                    else "missing"
                ),
                "prompt_trace": prompt_trace,
                "upstream": raw,
            }
            try:
                candidate = parse_candidate(completion, tools=shard)
                candidates = [candidate]
            except ValueError as exc:
                trace["error"] = str(exc)
                trace["schema_valid"] = False
                try:
                    candidates = parse_transport_candidates(completion, tools=shard)
                except ValueError as transport_exc:
                    trace["transport_error"] = str(transport_exc)
                    candidates = []
                else:
                    trace["transport_candidates"] = [asdict(item) for item in candidates]
            else:
                trace["schema_valid"] = True
                trace["candidate"] = asdict(candidate)
            if status != 200 and "error" not in trace:
                trace["error"] = f"upstream HTTP status {status}"
                candidates = []
            return trace, candidates

        with ThreadPoolExecutor(max_workers=min(max(1, self.config.batch_size), len(shards))) as executor:
            futures = [executor.submit(ask, shard) for shard in shards]
            for future in as_completed(futures):
                trace, candidates = future.result()
                candidate_traces.append(trace)
                if candidates:
                    if trace.get("schema_valid") is True:
                        valid_candidates.extend(candidates)
                    else:
                        # This is transport-only recovery.  It is not a
                        # correctness decision and must not be described as a
                        # schema-valid candidate in the trace.
                        transport_candidates.extend(candidates)
        candidate_traces.sort(key=lambda item: str(item.get("tools", [{}])[0].get("name", "")))

        aggregate_trace: dict[str, Any] = {}
        selected_candidates: list[Candidate] = []
        fallback_used = False
        transport_fallback_used = False
        aggregate_completion = ""
        if valid_candidates:
            aggregate, aggregate_prompt_trace = _aggregate_prompt(
                valid_candidates,
                tools,
                routed_messages,
                config=self.config,
            )
            if aggregate_prompt_trace["prompt_over_budget"]:
                aggregate_trace = {
                    "prompt": aggregate,
                    "prompt_trace": aggregate_prompt_trace,
                    "error": "aggregate prompt over budget",
                }
            else:
                payload = self._upstream_payload(source, aggregate, max_tokens=self.config.aggregate_max_tokens)
                status, _headers, body, raw = self._request_upstream(payload)
                aggregate_completion = self._completion_text(body)
                aggregate_trace = {
                    "prompt": aggregate,
                    "prompt_trace": aggregate_prompt_trace,
                    "completion": aggregate_completion,
                    "upstream": raw,
                    "status": status,
                }
                try:
                    selected_candidates = parse_candidates(aggregate_completion, tools=tools)
                except ValueError as exc:
                    aggregate_trace["error"] = str(exc)
        if not selected_candidates and self.config.fallback_to_highest_confidence and valid_candidates:
            selected_candidates = [max(valid_candidates, key=lambda item: item.confidence)]
            fallback_used = True
        if not selected_candidates and transport_candidates:
            # Preserve a syntactically identifiable call for the evaluator.
            # Do not validate, repair, or reinterpret its name/arguments here.
            selected_candidates = list(transport_candidates)
            transport_fallback_used = True

        model = str(source.get("model") or "")
        usage = _sum_upstream_usage(
            [
                *(trace.get("upstream") for trace in candidate_traces),
                aggregate_trace.get("upstream"),
            ]
        )
        route_trace: dict[str, Any] = {
            "mode": "parallel_candidate",
            "config": asdict(self.config),
            "context": context_trace,
            "candidate_count": len(valid_candidates),
            "transport_candidate_count": len(transport_candidates),
            "candidate_shards": candidate_traces,
            "aggregate": aggregate_trace,
            "selected": (
                asdict(selected_candidates[0])
                if len(selected_candidates) == 1
                else [asdict(item) for item in selected_candidates]
                if selected_candidates
                else None
            ),
            "selected_candidates": [asdict(item) for item in selected_candidates],
            "fallback_used": fallback_used,
            "transport_fallback_used": transport_fallback_used,
        }
        if usage is not None:
            route_trace["usage"] = usage
        native_response: dict[str, Any] | None = None
        if not selected_candidates and self.config.fallback_to_native_chat:
            native_response, native_trace = self._native_chat_fallback(source)
            route_trace["native_fallback_used"] = native_response is not None
            route_trace["native_fallback"] = native_trace
        else:
            route_trace["native_fallback_used"] = False
        if not selected_candidates:
            if native_response is not None:
                return native_response, route_trace
            content = aggregate_completion or (candidate_traces[0].get("completion", "") if candidate_traces else "")
            response = {
                "id": f"parallel-candidate-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content, "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
            }
            response["usage"] = _response_usage(usage)
            return response, route_trace

        response = {
            "id": f"parallel-candidate-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{uuid.uuid4().hex}",
                                "type": "function",
                                "function": {
                                    "name": candidate.name,
                                    "arguments": _candidate_arguments_text(candidate),
                                },
                            }
                            for candidate in selected_candidates
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        response["usage"] = _response_usage(usage)
        return response, route_trace

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None


__all__ = [
    "Candidate",
    "ParallelCandidateConfig",
    "ParallelCandidateProxy",
    "parse_candidate",
    "parse_candidates",
    "parse_transport_candidate",
    "parse_transport_candidates",
]
