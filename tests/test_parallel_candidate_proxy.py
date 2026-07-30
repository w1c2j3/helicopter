from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from helicopter_cli.parallel_candidate_proxy import (
    Candidate,
    ParallelCandidateConfig,
    ParallelCandidateProxy,
    _aggregate_prompt,
    _candidate_prompt,
    parse_candidate,
)
from helicopter_cli.rwkv_agent_prompt import (
    LongContextConfig,
    RWKV_FLOWER_JSON_PROMPT_STYLE,
    build_rwkv_json_call_prompt,
    compact_messages_for_long_context,
    normalize_messages,
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

    search_tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search.",
                "parameters": {
                    "type": "object",
                    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    with pytest.raises(ValueError, match=r"queries\[0\] must be string"):
        parse_candidate(
            '{"name":"search","arguments":{"queries":[{"query":"wrong shape"}]}}',
            tools=search_tools,
        )


@pytest.mark.parametrize(
    "completion",
    [
        '</think>{"name":"bash","arguments":{"command":"echo hi"}}',
        '</think>```json\n{"name":"bash","arguments":{"command":"echo hi"}}\n```',
        'reasoning first\n</think>```json\n{"name":"bash","arguments":{"command":"echo hi"}}\n```',
    ],
)
def test_parse_candidate_accepts_json_after_explicit_think_close(completion: str) -> None:
    candidate = parse_candidate(completion, tools=TOOLS)

    assert candidate.name == "bash"
    assert candidate.arguments == {"command": "echo hi"}


def test_parse_candidate_accepts_one_strict_native_tool_call_envelope() -> None:
    candidate = parse_candidate(
        '{"tool_calls":[{"id":"call_1","type":"function",'
        '"function":{"name":"bash","arguments":"{\\"command\\":\\"echo hi\\"}"}}]}',
        tools=TOOLS,
    )

    assert candidate.name == "bash"
    assert candidate.arguments == {"command": "echo hi"}


def test_parse_candidate_rejects_multiple_native_tool_calls_without_selecting_one() -> None:
    with pytest.raises(ValueError, match="exactly one call"):
        parse_candidate(
            '{"tool_calls":['
            '{"function":{"name":"bash","arguments":"{\\"command\\":\\"pwd\\"}"}},'
            '{"function":{"name":"submit","arguments":"{\\"answer\\":\\"done\\"}"}}'
            ']}',
            tools=TOOLS,
        )


def test_candidate_prompt_compacts_large_tool_schema() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "large_tool",
                "description": "tool description " * 400,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "parameter description " * 400,
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ]
    prompt, trace = _candidate_prompt(
        tools,
        [{"role": "user", "content": "Run the requested action."}],
        config=ParallelCandidateConfig(prompt_max_chars=2048, context_chars=256),
    )

    assert trace["prompt_over_budget"] is False
    assert len(prompt) < 2048
    assert prompt.count("tool description") < 20
    assert prompt.count("parameter description") < 20
    assert '"command"' in prompt


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
    assert "Assistant: <think></think>\n```json" in prompt
    assert prompt.endswith("Assistant: <think></think>\n```json\n")
    assert prompt_trace["prompt_chars"] == len(prompt)


def test_normalize_messages_recovers_evalscope_chat_message_repr() -> None:
    malformed_wire_content = (
        "[ChatMessageUser(id='abc123', content='Find the answer\\nfrom the corpus.', "
        "source=None, metadata=None, internal=None, perf_metrics=None, role='user', "
        "tool_call_id=None)]"
    )
    source = [{"role": "user", "content": malformed_wire_content}]

    normalized = normalize_messages(source)

    assert normalized == [{"role": "user", "content": "Find the answer\nfrom the corpus."}]
    assert source[0]["content"] == malformed_wire_content

    embedded = [{"role": "user", "content": "Question: " + malformed_wire_content + "\nKeep this."}]
    assert normalize_messages(embedded) == [
        {"role": "user", "content": "Question: Find the answer\nfrom the corpus.\nKeep this."}
    ]


def test_parallel_candidate_route_preserves_recovered_agent_question(tmp_path: Path) -> None:
    malformed_wire_content = (
        "[ChatMessageUser(id='abc123', content='Find the answer from the corpus.', "
        "source=None, metadata=None, internal=None, perf_metrics=None, role='user', "
        "tool_call_id=None)]"
    )
    seen_prompts: list[str] = []
    proxy = ParallelCandidateProxy(
        "http://127.0.0.1:1/v1",
        api_key="secret",
        trace_path=tmp_path / "parallel-repr.jsonl",
    )

    def fake_request(payload: dict[str, object]) -> tuple[int, dict[str, str], dict[str, object], dict[str, object]]:
        seen_prompts.append(str(payload["prompt"]))
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"name":"bash","arguments":{"command":"grep answer corpus"},"confidence":0.9}',
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        return 200, {}, body, {}

    proxy._request_upstream = fake_request  # type: ignore[method-assign]
    result, trace = proxy._route(
        {
            "model": "rwkv",
            "messages": [{"role": "user", "content": malformed_wire_content}],
            "tools": [TOOLS[0]],
            "tool_choice": "auto",
            "max_tokens": 2048,
        }
    )

    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert trace["candidate_count"] == 1
    assert seen_prompts
    assert "User: Find the answer from the corpus." in seen_prompts[0]
    assert "ChatMessageUser" not in seen_prompts[0]


def test_rwkv_flower_json_prompt_uses_g1h_nocot_transcript() -> None:
    prompt, _ = build_rwkv_json_call_prompt(
        "Keep the original system instruction.",
        [
            {"role": "system", "content": "Source system instruction."},
            {"role": "user", "content": "Run the requested action."},
        ],
        history_max_chars=600,
        prompt_max_chars=1200,
        prompt_style=RWKV_FLOWER_JSON_PROMPT_STYLE,
    )

    assert "User\u273fSystem:\nKeep the original system instruction.\u273f" in prompt
    assert "User\u273fSystem:\nSource system instruction.\u273f" in prompt
    assert "User\u273fRun the requested action.\u273f" in prompt
    assert prompt.endswith("Bot\u273f<think></think>\n```json\n")


def test_aggregate_prompt_bounds_large_tool_catalog_after_candidate_validation() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{index}",
                "description": "A long tool description. " * 80,
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
        for index in range(40)
    ]
    prompt, trace = _aggregate_prompt(
        [Candidate(name="tool_0", arguments={"value": "ok"}, confidence=0.9, evidence="validated")],
        tools,
        [{"role": "user", "content": "Run the best available action."}],
        config=ParallelCandidateConfig(prompt_max_chars=2048, context_chars=256),
    )

    assert trace["prompt_over_budget"] is False
    assert "Valid tool names:" in prompt
    assert '"tool_0"' in prompt
    assert '"properties"' not in prompt


def test_aggregate_prompt_limits_candidates_by_confidence() -> None:
    prompt, _trace = _aggregate_prompt(
        [
            Candidate(name="bash", arguments={"command": "echo high"}, confidence=0.9, evidence="high"),
            Candidate(name="submit", arguments={"answer": "low"}, confidence=0.1, evidence="low"),
        ],
        TOOLS,
        [{"role": "user", "content": "Run the requested action."}],
        config=ParallelCandidateConfig(max_candidates=1),
    )

    assert '"command":"echo high"' in prompt
    assert '"answer":"low"' not in prompt


def test_parallel_candidate_proxy_returns_validated_tool_call_and_trace(tmp_path) -> None:
    received: list[dict[str, object]] = []
    received_paths: list[str] = []

    class UpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(body)
            received.append(payload)
            received_paths.append(self.path)
            prompt = payload["prompt"].lower()
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
                            "text": content,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
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
    assert result["usage"] == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}
    assert received
    assert all(payload["prompt"].endswith("Assistant: <think></think>\n```json\n") for payload in received)
    assert all("tools" not in payload for payload in received)
    assert all("messages" not in payload for payload in received)
    assert all("Keep this system message unchanged." in payload["prompt"] for payload in received)
    assert all("Conversation transcript JSON:" not in payload["prompt"] for payload in received)
    assert all("Assistant: <think></think>\n```json" in payload["prompt"] for payload in received)
    assert all("Bot\u273f<think></think>\n```json" not in payload["prompt"] for payload in received)
    assert all(
        "System: Keep this system message unchanged."
        in payload["prompt"]
        for payload in received
    )
    assert all("User: Run the requested command." in payload["prompt"] for payload in received)
    assert received_paths
    assert all(path == "/v1/completions" for path in received_paths)

    record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert record["request"]["json"]["messages"][0]["content"] == "Keep this system message unchanged."
    assert record["router"]["mode"] == "parallel_candidate"
    assert record["router"]["candidate_count"] == 1
    assert record["response"]["body"]["choices"][0]["message"]["tool_calls"]
    assert record["response"]["body"]["usage"] == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}


def test_parallel_candidate_proxy_trace_encodes_bytes_without_breaking_response(tmp_path) -> None:
    trace = tmp_path / "parallel-bytes.jsonl"
    proxy = ParallelCandidateProxy(
        "http://127.0.0.1:1/v1",
        api_key="secret",
        trace_path=trace,
    )
    proxy._route = lambda _source: (  # type: ignore[method-assign]
        {"choices": []},
        {"mode": "test", "raw": b"\x00\xff"},
    )
    try:
        proxy.start()
        request = Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps({"model": "rwkv", "messages": [], "tools": TOOLS}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:  # noqa: S310 - test-only local server
            assert response.status == 200
            assert json.loads(response.read()) == {"choices": []}
    finally:
        proxy.close()

    record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert record["router"]["raw"] == {"__type__": "bytes", "base64": "AP8="}


def test_parallel_candidate_proxy_direct_route_parses_upstream_json() -> None:
    proxy = ParallelCandidateProxy("http://127.0.0.1:1/v1", api_key="secret", trace_path=Path("trace.jsonl"))
    proxy._forward = lambda *_args, **_kwargs: (  # type: ignore[method-assign]
        200,
        {"Content-Type": "application/json"},
        b'{"id":"upstream","model":"rwkv","choices":[]}',
    )

    result, route_trace = proxy._route({"model": "rwkv", "messages": []})

    assert result["model"] == "rwkv"
    assert result["id"] == "upstream"
    assert route_trace["mode"] == "direct"
    assert route_trace["upstream_response"]["model"] == "rwkv"


def test_upstream_candidate_payload_has_flower_response_stops() -> None:
    proxy = ParallelCandidateProxy("http://127.0.0.1:1/v1", api_key="secret", trace_path=Path("trace.jsonl"))

    payload = proxy._upstream_payload(
        {"model": "rwkv", "temperature": 0.0},
        "Bot\u273f<think></think>\n```json\n",
        max_tokens=2048,
    )

    assert payload["prompt"] == "Bot\u273f<think></think>\n```json\n"
    assert "messages" not in payload
    assert payload["stop"][:2] == ["\n```", "```"]
    assert "\u273f" in payload["stop"]
    preserved = proxy._upstream_payload(
        {"model": "rwkv", "stop": ["CUSTOM"]},
        "prompt",
        max_tokens=128,
    )
    assert preserved["stop"] == ["CUSTOM"]
    assert preserved["prompt"] == "prompt"
