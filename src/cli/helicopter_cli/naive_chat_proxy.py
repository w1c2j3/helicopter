"""Small local OpenAI-compatible proxy for the RWKV naive Chat boundary."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .evalscope_agent_compat import adapt_tool_call_response
from .naive_chat import serialize_openai_request


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: ("Bearer [redacted]" if key.lower() == "authorization" else value)
        for key, value in headers.items()
    }


def _json_or_text(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body.decode("utf-8", "replace")


class NaiveChatProxy:
    """Forward OpenAI traffic after converting chat messages to naive Chat.

    The proxy is deliberately process-local to a Helicopter evaluation run.
    Every upstream request/response pair is appended to JSONL so raw model
    output remains available even when EvalScope reports an evaluation error.
    """

    def __init__(self, upstream_base_url: str, *, api_key: str, trace_path: Path) -> None:
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.api_key = api_key
        self.trace_path = trace_path
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("naive Chat proxy is not started")
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def start(self) -> str:
        if self._server is not None:
            return self.base_url
        trace_path = self.trace_path
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _handle(self) -> None:
                started = perf_counter()
                request_headers = {key: value for key, value in self.headers.items()}
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                source_payload: Any = _json_or_text(body)
                forwarded_payload: Any = source_payload
                response_status = 502
                response_headers: dict[str, str] = {}
                response_body = b""
                upstream_response_body: Any = None
                compatibility: dict[str, Any] = {"status": "not_run", "reason": None}
                error: dict[str, str] | None = None
                is_chat_request = self.command == "POST" and self.path.split("?", 1)[0].endswith("/chat/completions")
                upstream_path = "/v1/completions" if is_chat_request else self.path
                try:
                    if is_chat_request:
                        if not isinstance(source_payload, dict):
                            raise ValueError("chat completion body must be a JSON object")
                        forwarded_payload = serialize_openai_request(source_payload)
                        outgoing_body = json.dumps(forwarded_payload, ensure_ascii=False).encode("utf-8")
                    else:
                        outgoing_body = body

                    upstream_url = proxy._upstream_url(upstream_path)
                    outgoing_headers = {
                        "Content-Type": self.headers.get("Content-Type", "application/json"),
                        "Accept": self.headers.get("Accept", "application/json"),
                        "Authorization": f"Bearer {proxy.api_key}",
                    }
                    response = proxy._request(upstream_url, outgoing_body, outgoing_headers)
                    response_status, response_headers, response_body = response
                    if is_chat_request and response_status == 200:
                        upstream_response_body = _json_or_text(response_body)
                        response_body = proxy._adapt_completion_response(response_body)
                        chat_payload = _json_or_text(response_body)
                        adapted_payload, compatibility = adapt_tool_call_response(
                            chat_payload,
                            tools=source_payload.get("tools")
                            if isinstance(source_payload, dict) and isinstance(source_payload.get("tools"), list)
                            else None,
                        )
                        response_body = json.dumps(adapted_payload, ensure_ascii=False).encode("utf-8")
                except HTTPError as exc:
                    response_status = exc.code
                    response_headers = {key: value for key, value in exc.headers.items()}
                    response_body = exc.read()
                    error = {"type": "HTTPError", "message": str(exc)}
                except (OSError, URLError, ValueError) as exc:
                    response_status = 502
                    response_body = json.dumps(
                        {"error": {"type": type(exc).__name__, "message": str(exc)}}
                    ).encode("utf-8")
                    response_headers = {"Content-Type": "application/json"}
                    error = {"type": type(exc).__name__, "message": str(exc)}

                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "method": self.command,
                    "path": self.path,
                    "request": {
                        "headers": _redact_headers(request_headers),
                        "json": source_payload,
                    },
                    "forwarded_request": {
                        "url": proxy._upstream_url(upstream_path),
                        "headers": {"Authorization": "Bearer [redacted]", "Content-Type": "application/json"},
                        "json": forwarded_payload,
                    },
                    "response": {
                        "status": response_status,
                        "headers": response_headers,
                        "body": _json_or_text(response_body),
                        "upstream_body_before_compatibility": upstream_response_body,
                    },
                    "compatibility": compatibility,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "error": error,
                }
                with proxy._lock:
                    with trace_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

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
        self._thread = threading.Thread(target=self._server.serve_forever, name="helicopter-naive-chat", daemon=True)
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

    @staticmethod
    def _adapt_completion_response(body: bytes) -> bytes:
        """Wrap raw completion text in the chat response shape EvalScope expects."""

        payload = _json_or_text(body)
        if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
            return body
        changed = False
        choices: list[Any] = []
        for choice in payload["choices"]:
            if not isinstance(choice, dict) or "text" not in choice:
                choices.append(choice)
                continue
            adapted = dict(choice)
            message = adapted.get("message")
            message = dict(message) if isinstance(message, dict) else {}
            message.setdefault("role", "assistant")
            message["content"] = adapted.pop("text") or ""
            adapted["message"] = message
            choices.append(adapted)
            changed = True
        if not changed:
            return body
        output = dict(payload)
        output["choices"] = choices
        return json.dumps(output, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _request(url: str, body: bytes, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        request = Request(url, data=body if body else None, headers=headers, method="POST" if body else "GET")
        with urlopen(request, timeout=180) as response:  # noqa: S310 - configured local endpoint
            return response.status, dict(response.headers.items()), response.read()

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None


__all__ = ["NaiveChatProxy"]
