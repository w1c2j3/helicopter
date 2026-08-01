from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI


def main() -> int:
    prediction_path = Path("/home/rwkv/chase/EvalScope/results/evalscope/agent-local-k2-verifier-7p2b-20260731/predictions/rwkv7-g1h-7.2b-20260710-ctx10240/k2_verifier_k2_thinking.jsonl")
    row = json.loads(prediction_path.read_text(encoding="utf-8").splitlines()[0])
    messages = []
    for message in row["messages"]:
        item = {"role": message["role"], "content": message.get("content", "")}
        if message.get("tool_calls"):
            item["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id"):
            item["tool_call_id"] = message["tool_call_id"]
        messages.append(item)
    metadata = row["metadata"]
    client = OpenAI(base_url="http://127.0.0.1:29572/v1", api_key="rwkv-skills", timeout=120)
    response = client.chat.completions.create(
        model="rwkv7-g1h-7.2b-20260710-ctx10240",
        messages=messages,
        tools=metadata["tools"],
        tool_choice="auto",
        max_tokens=64,
        temperature=0.0,
    )
    print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
