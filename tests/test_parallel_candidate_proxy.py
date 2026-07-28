from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from helicopter_cli.parallel_candidate_proxy import (
    ParallelCandidateConfig,
    ParallelCandidateProxy,
    parse_candidate,
)
from helicopter_cli.rwkv_agent_prompt import (
    LongContextConfig,
    build_rwkv_json_call_prompt,
    compact_messages_for_long_context,
    trim_message_history,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the final answer.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def test_parse_candidate_is_strict_and_schema_bound() -> None:
    candidate = parse_candidate(
        '{"name":"bash","arguments":{"command":"echo hi"},"confidence":0.9,"evidence":"user asked to run it"}',
        tools=TOOLS,
    )
    assert candidate.name == "bash"
    assert candidate.arguments == {"command": "echo hi"}

    transport_wrapped = parse_candidate(
        '{"name":"bash","arguments":"{\\"command\\":\\"echo hi\\"}","id":"call_1"}',
        tools=TOOLS,
    )
    assert transport_wrapped.arguments == {"command": "echo hi"}

    with pytest.raises(ValueError, match="unknown fields"):
        parse_candidate('{"name":"bash","arguments":{"command":"echo hi","invented":1}}', tools=TOOLS)
    with pytest.raises(ValueError, match="missing required"):
        parse_candidate('{"name":"bash","arguments":{}}', tools=TOOLS)
    with pytest.raises(ValueError, match="not in the supplied tools"):
        parse_candidate('{"name":"python","arguments":{}}', tools=TOOLS)
    with pytest.raises(ValueError, match="must start"):
        parse_candidate('reasoning first\n{"name":"bash","arguments":{"command":"echo hi"}}', tools=TOOLS)


def test_rwkv_prompt_uses_role_transcript_and_newest_history_budget() -> None:
    messages = [
        {"role": "system", "content": "Original system prompt: never change this text."},
        {"role": "user", "content": "old context " + "x" * 100},
        {"role": "assistant", "content": '{"name":"bash","arguments":{"command":"pwd"}}'},
        {"role": "tool", "content": "tool output: " + "y" * 80},
        {"role": "user", "content": "run the final command"},
    ]
    bounded, truncated = trim_message_history(messages, max_chars=340)
    assert truncated is True
    assert bounded[-1]["content"] == "run the final command"
    assert any("truncated" in item["content"] for item in bounded)

    compacted, trace = compact_messages_for_long_context(
        messages,
        config=LongContextConfig(min_long_text_chars=80, max_chunk_chars=30, max_evidence_chars=90),
    )
    assert trace["compacted_message_count"] >= 1
    assert "Long document compacted" in compacted[1]["content"]

    prompt, prompt_trace = build_rwkv_json_call_prompt(
        "Router instructions",
        compacted,
        history_max_chars=600,
        prompt_max_chars=1200,
    )
    assert "System: Router instructions" in prompt
    assert "System: Original system prompt: never change this text." in prompt
    assert "User: Function output:" in prompt
    assert "Assistant: ```json" in prompt
    assert prompt.endswith("Assistant: ```json\n")
    assert prompt_trace["prompt_chars"] == len(prompt)


def test_parallel_candidate_proxy_returns_validated_tool_call_and_trace(tmp_path) -> None:
    received: list[dict[str, object]] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(body)
            received.append(payload)
            prompt = payload["messages"][0]["content"].lower()
            if "aggregator for a parallel" in prompt:
                content = '{"name":"bash","arguments":{"command":"echo hi"},"confidence":0.95,"evidence":"validated candidate"}'
            elif '"name":"bash"' in prompt:
                content = '{"name":"bash","arguments":{"command":"echo hi"},"confidence":0.8,"evidence":"user request"}'
            else:
                content = "not a candidate"
            response = json.dumps(
                {
                    "id": "upstream",
                    "model": payload["model"],
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    trace = tmp_path / "parallel.jsonl"
    proxy = ParallelCandidateProxy(
        f"http://127.0.0.1:{upstream.server_port}/v1",
        api_key="secret",
        trace_path=trace,
        config=ParallelCandidateConfig(chunk_tools=1, batch_size=2, fallback_to_highest_confidence=False),
    )
    try:
        proxy.start()
        source = {
            "model": "rwkv",
            "messages": [
                {"role": "system", "content": "Keep this system message unchanged."},
                {"role": "user", "content": "Run the requested command."},
            ],
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 512,
        }
        request = Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps(source).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer client-secret"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:  # noqa: S310 - test-only local server
            result = json.loads(response.read())
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert tool_calls[0]["function"]["name"] == "bash"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"command": "echo hi"}
    assert received
    assert all("tools" not in payload for payload in received)
    assert all("Keep this system message unchanged." in payload["messages"][0]["content"] for payload in received)
    assert all("Conversation transcript JSON:" not in payload["messages"][0]["content"] for payload in received)
    assert all("Assistant: ```json" in payload["messages"][0]["content"] for payload in received)
    assert all("User: Run the requested command." in payload["messages"][0]["content"] for payload in received)

    record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert record["request"]["json"]["messages"][0]["content"] == "Keep this system message unchanged."
    assert record["router"]["mode"] == "parallel_candidate"
    assert record["router"]["candidate_count"] == 1
    assert record["response"]["body"]["choices"][0]["message"]["tool_calls"]
