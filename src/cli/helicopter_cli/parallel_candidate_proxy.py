"""Local Agent proxy for RWKV endpoints without native OpenAI tool parsing.

The proxy is deliberately an evaluation-layer adapter.  It does not alter the
EvalScope task, system messages, or tool semantics.  It embeds the original
conversation verbatim in short routing prompts, asks the upstream model for
strict JSON candidates without sending OpenAI ``tools`` fields, and exposes a
validated candidate as a normal OpenAI ``message.tool_calls`` response.

Every source request, candidate request/response, aggregate request/response,
selection decision, and final response is written to JSONL.  A malformed or
unvalidated candidate is never turned into a tool call.
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

from .rwkv_agent_prompt import (
    DEFAULT_LONG_DOC_MAX_CHARS,
    DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS,
    DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS,
    DEFAULT_LONG_DOC_MIN_CHARS,
    DEFAULT_LONG_DOC_OVERLAP_LINES,
    LongContextConfig,
    RWKV_FLOWER_JSON_PROMPT_STYLE,
    build_rwkv_json_call_prompt,
    compact_messages_for_long_context,
    normalize_messages,
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


# Match the RWKV NoCoT function-calling prefill used by rwkv-skills.  The
# empty think block is part of the request-time serialization only; source
# messages and their semantics remain unchanged.
_FLOWER_NO_COT_JSON_ASSISTANT_PREFIX = "Bot\u273f<think></think>\n```json\n"
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


def _json_object(text: str) -> dict[str, Any]:
    """Extract one complete model-generated JSON object without repairing it.

    RWKV may emit a reasoning segment followed by an explicit ``</think>``
    delimiter and the requested JSON object.  The delimiter is part of the
    model output; using only the suffix after that delimiter keeps extraction
    deterministic while preserving strict schema validation below.
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
    if not source.startswith("{"):
        raise ValueError("completion must start with a JSON object")
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
    if not isinstance(value, dict):
        raise ValueError("completion JSON value must be an object")
    return value


def parse_candidate(text: str, *, tools: list[Any]) -> Candidate:
    """Strictly validate a candidate against the supplied tool schemas."""

    value = _json_object(text)
    unknown = set(value).difference({"name", "arguments", "confidence", "evidence", "id", "tool_call_id"})
    if unknown:
        raise ValueError(f"candidate contains unknown fields: {sorted(unknown)}")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("candidate name must be a non-empty string")
    name = name.strip()
    schemas = _schema_by_name(tools)
    if name not in schemas:
        raise ValueError(f"candidate name {name!r} is not in the supplied tools")
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


def _compact_prompt_tools(tools: list[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for schema in _schemas(tools):
        output.append(
            {
                "name": schema["name"],
                "description": schema["description"][:500],
                "parameters": schema["parameters"],
            }
        )
    return output


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
        assistant_prefix=_FLOWER_NO_COT_JSON_ASSISTANT_PREFIX,
        prompt_style=RWKV_FLOWER_JSON_PROMPT_STYLE,
    )


def _aggregate_prompt(
    candidates: list[Candidate],
    tools: list[Any],
    messages: list[dict[str, str]],
    *,
    config: ParallelCandidateConfig,
) -> tuple[str, dict[str, Any]]:
    rows = [
        {
            "name": item.name,
            "arguments": item.arguments,
            "confidence": item.confidence,
            "evidence": item.evidence,
        }
        for item in sorted(candidates, key=lambda item: item.confidence, reverse=True)
    ]
    system_prompt = "\n".join(
        [
            "You are the aggregator for a parallel candidate tool-call router.",
            "Choose the best next action from the candidates for the original conversation.",
            "Return exactly one JSON object with only these fields:",
            '{"name":"tool_name","arguments":{},"confidence":0.0,"evidence":"short reason"}',
            "Use only a supplied tool name and keep the selected arguments unless the conversation proves they are wrong.",
            "Do not include id, type, tool_calls, function, analysis, markdown, or extra fields.",
            "Candidates:",
            json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
            "Valid tools:",
            json.dumps(_compact_prompt_tools(tools), ensure_ascii=False, separators=(",", ":")),
        ]
    )
    return build_rwkv_json_call_prompt(
        system_prompt,
        messages,
        history_max_chars=config.context_chars,
        prompt_max_chars=config.prompt_max_chars,
        assistant_prefix=_FLOWER_NO_COT_JSON_ASSISTANT_PREFIX,
        prompt_style=RWKV_FLOWER_JSON_PROMPT_STYLE,
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
            self._upstream_url("/v1/chat/completions"),
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
            "url": self._upstream_url("/v1/chat/completions"),
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

    @staticmethod
    def _completion_text(response: Any) -> str:
        if not isinstance(response, Mapping):
            return ""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            return ""
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
        payload["messages"] = [{"role": "user", "content": prompt}]
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

        def ask(shard: list[dict[str, Any]]) -> tuple[dict[str, Any], Candidate | None]:
            prompt, prompt_trace = _candidate_prompt(shard, routed_messages, config=self.config)
            if prompt_trace["prompt_over_budget"]:
                return {
                    "tools": shard,
                    "prompt": prompt,
                    "completion": "",
                    "finish_reason": "prompt_over_budget",
                    "prompt_trace": prompt_trace,
                    "error": f"candidate prompt length {len(prompt)} exceeds {self.config.prompt_max_chars}",
                }, None
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
            except ValueError as exc:
                trace["error"] = str(exc)
                candidate = None
            else:
                trace["candidate"] = asdict(candidate)
            if status != 200 and "error" not in trace:
                trace["error"] = f"upstream HTTP status {status}"
                candidate = None
            return trace, candidate

        with ThreadPoolExecutor(max_workers=min(max(1, self.config.batch_size), len(shards))) as executor:
            futures = [executor.submit(ask, shard) for shard in shards]
            for future in as_completed(futures):
                trace, candidate = future.result()
                candidate_traces.append(trace)
                if candidate is not None:
                    valid_candidates.append(candidate)
        candidate_traces.sort(key=lambda item: str(item.get("tools", [{}])[0].get("name", "")))

        aggregate_trace: dict[str, Any] = {}
        selected: Candidate | None = None
        fallback_used = False
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
                    selected = parse_candidate(aggregate_completion, tools=tools)
                except ValueError as exc:
                    aggregate_trace["error"] = str(exc)
        if selected is None and self.config.fallback_to_highest_confidence and valid_candidates:
            selected = max(valid_candidates, key=lambda item: item.confidence)
            fallback_used = True

        model = str(source.get("model") or "")
        route_trace: dict[str, Any] = {
            "mode": "parallel_candidate",
            "config": asdict(self.config),
            "context": context_trace,
            "candidate_count": len(valid_candidates),
            "candidate_shards": candidate_traces,
            "aggregate": aggregate_trace,
            "selected": asdict(selected) if selected is not None else None,
            "fallback_used": fallback_used,
        }
        if selected is None:
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
                                    "name": selected.name,
                                    "arguments": json.dumps(selected.arguments, ensure_ascii=False, separators=(",", ":")),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
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


__all__ = ["Candidate", "ParallelCandidateConfig", "ParallelCandidateProxy", "parse_candidate"]
