from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from helicopter_cli.naive_chat_proxy import NaiveChatProxy
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
    transcript = forwarded["prompt"]

    assert forwarded["model"] == "rwkv"
    assert forwarded["temperature"] == 0
    assert forwarded.keys() >= {"model", "prompt", "temperature", "stream"}
    assert "messages" not in forwarded
    assert not ({"tools", "tool_choice", "parallel_tool_calls"} & forwarded.keys())
    assert transcript.endswith("Assistant: <think></think>\n```json\n")
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


def test_proxy_serializes_request_preserves_response_and_records_trace(tmp_path) -> None:
    received: list[dict[str, object]] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            received.append(json.loads(body))
            response = b'{"id":"upstream","choices":[{"text":"raw model output","finish_reason":"stop"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    trace = tmp_path / "trace.jsonl"
    proxy = NaiveChatProxy(f"http://127.0.0.1:{upstream.server_port}/v1", api_key="secret", trace_path=trace)
    try:
        proxy.start()
        request = Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": "rwkv",
                    "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                    "tools": [{"type": "function", "function": {"name": "bash"}}],
                    "tool_choice": "auto",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer client-secret"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:  # noqa: S310 - test-only local server
            assert response.status == 200
            response_body = json.loads(response.read())
            assert response_body["choices"][0]["message"] == {
                "role": "assistant",
                "content": "raw model output",
            }
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert received[0]["prompt"].endswith("Assistant: <think></think>\n```json\n")
    assert "messages" not in received[0]
    assert "tools" not in received[0]
    trace_record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert trace_record["response"]["body"]["choices"][0]["message"]["content"] == "raw model output"
    assert trace_record["forwarded_request"]["url"].endswith("/v1/completions")
    assert trace_record["request"]["headers"]["Authorization"] == "Bearer [redacted]"


def test_proxy_adapts_bfcl_content_to_native_tool_calls(tmp_path) -> None:
    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            response = (
                b'{"id":"upstream","choices":[{"text":"[{\\"bash\\":{\\"command\\":\\"echo hi\\"}}]",'
                b'"finish_reason":"stop"}]}'
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    trace = tmp_path / "compat-trace.jsonl"
    proxy = NaiveChatProxy(f"http://127.0.0.1:{upstream.server_port}/v1", api_key="secret", trace_path=trace)
    request_body = {
        "model": "rwkv",
        "messages": [{"role": "user", "content": "Run it."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
    try:
        proxy.start()
        request = Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:  # noqa: S310 - test-only local server
            payload = json.loads(response.read())
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    message = payload["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"command": "echo hi"}
    trace_record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert trace_record["compatibility"]["status"] == "converted"
    assert trace_record["response"]["upstream_body_before_compatibility"]["choices"][0]["text"]
