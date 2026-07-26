from __future__ import annotations

import argparse
import json
import os
from http.client import HTTPConnection
from pathlib import Path
from time import perf_counter, strftime, gmtime


MODEL = "rwkv7-g1h-2.9b-20260710-ctx10240"
BASE_URL = "http://127.0.0.1:19329/v1"


def payload(*, naive_chat: bool = False) -> dict[str, object]:
    if naive_chat:
        return {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "System: You are solving a geometry question.\n\n"
                        "User: Find the area of a triangle with a base of 10 units and height of 5 units.\n\n"
                        "Assistant:"
                    ),
                }
            ],
            "temperature": 0,
            "max_tokens": 768,
        }
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are solving a function-calling benchmark. Choose the tool call or calls needed "
                    "to satisfy the user request. Return calls through the tool calling interface."
                ),
            },
            {
                "role": "user",
                "content": "User: Find the area of a triangle with a base of 10 units and height of 5 units.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate_triangle_area",
                    "description": "Calculate the area of a triangle given its base and height.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "base": {"type": "integer", "description": "The base of the triangle."},
                            "height": {"type": "integer", "description": "The height of the triangle."},
                            "unit": {
                                "type": "string",
                                "description": "The unit of measure (defaults to 'units' if not specified)",
                            },
                        },
                        "required": ["base", "height"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "temperature": 0,
        "max_tokens": 768,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--naive-chat", action="store_true")
    args = parser.parse_args()

    request_payload = payload(naive_chat=args.naive_chat)
    started = perf_counter()
    record: dict[str, object] = {
        "request": {
            "method": "POST",
            "url": f"{BASE_URL}/chat/completions",
            "headers": {"Authorization": "Bearer [redacted]", "Content-Type": "application/json"},
            "json": request_payload,
        },
        "started_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": None,
        "response_headers": {},
        "response_body": None,
        "error": None,
    }
    try:
        connection = HTTPConnection("127.0.0.1", 19329, timeout=60)
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(request_payload, ensure_ascii=False),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('HELICOPTER_EVAL_API_KEY', '')}",
            },
        )
        response = connection.getresponse()
        record["status"] = response.status
        record["response_headers"] = dict(response.getheaders())
        body = response.read().decode("utf-8", "replace")
        try:
            record["response_body"] = json.loads(body)
        except json.JSONDecodeError:
            record["response_body"] = body
    except Exception as exc:  # noqa: BLE001 - preserve the baseline failure.
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        record["duration_ms"] = round((perf_counter() - started) * 1000, 2)

    rendered = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
