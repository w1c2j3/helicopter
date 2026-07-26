from __future__ import annotations

import argparse
import json
import os
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlsplit

from helicopter_cli.naive_chat_proxy import NaiveChatProxy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()

    proxy = NaiveChatProxy(
        "http://127.0.0.1:19329/v1",
        api_key=os.environ.get("HELICOPTER_EVAL_API_KEY", ""),
        trace_path=args.trace,
    )
    proxy.start()
    try:
        body = {
            "model": "rwkv7-g1h-2.9b-20260710-ctx10240",
            "messages": [
                {"role": "system", "content": "Answer the user's geometry question."},
                {"role": "user", "content": "Find the area of a triangle with base 10 and height 5."},
            ],
            "tools": [{"type": "function", "function": {"name": "calculator", "parameters": {}}}],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
        }
        endpoint = urlsplit(proxy.base_url)
        connection = HTTPConnection(endpoint.hostname, endpoint.port, timeout=180)
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(body, ensure_ascii=False),
            headers={"Content-Type": "application/json", "Authorization": "Bearer proxy-client"},
        )
        response = connection.getresponse()
        result = {"status": response.status, "body": json.loads(response.read().decode("utf-8"))}
    finally:
        proxy.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
