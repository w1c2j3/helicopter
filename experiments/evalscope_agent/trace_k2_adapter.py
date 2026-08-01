from __future__ import annotations

import traceback

from evalscope.api.model import GenerateConfig
from evalscope.api.registry import get_benchmark
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.config import TaskConfig


def main() -> int:
    config = TaskConfig(
        model="rwkv7-g1h-7.2b-20260710-ctx10240",
        api_url="http://127.0.0.1:29572/v1",
        api_key="rwkv-skills",
        datasets=["k2_verifier"],
    )
    adapter = get_benchmark("k2_verifier", config)
    datasets, _ = adapter.load()
    sample = next(iter(datasets["k2_thinking"]))
    print("messages", len(sample.input), "tools", len(sample.tools or []))
    api = OpenAICompatibleAPI(
        model_name="rwkv7-g1h-7.2b-20260710-ctx10240",
        base_url="http://127.0.0.1:29572/v1",
        api_key="rwkv-skills",
        config=GenerateConfig(max_tokens=64, temperature=0.0, timeout=120, retries=0, retry_interval=0),
    )
    try:
        output = api.generate(sample.input, sample.tools or [], "auto", api.config)
        print("direct output", output.model_dump())
    except Exception:
        traceback.print_exc()
    try:
        output = adapter._on_inference(api, sample)
        print("adapter output", output.model_dump())
    except Exception:
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
