# LightEval RWKV completion profiles

These files are model-and-category profiles, not per-task benchmark files. A
profile groups all tasks that share one raw-input adapter, RWKV prompt wrapper,
rollout count, and sampling policy.

The active source path is:

```text
dataset row
  -> LightEval task prompt function (Doc.raw_query)
  -> [prompt].template
  -> LiteLLMClient source implementation
  -> POST /v1/completions
  -> choices[0:n]
  -> ModelResponse
  -> task-native LightEval metric/adapter
```

`[evaluation].num_samples` is the rollout count and is copied to both
`Doc.num_samples` and the HTTP request's `n`. All other request-time generation
fields come only from `[sampling]`. Fields not implemented by vllm-rwkv's
`CompletionRequest` (for example `stop_tokens`, `ban_tokens`, `pad_zero`, or
`prefill_chunk_size`) are rejected.

Example:

```bash
helicopter eval run g1h-7.2b \
  --config configs/eval/profiles/g1h-7.2b/math-cot.toml \
  --no-server
```

Pass one task from `[profile].tasks` as the optional positional task argument
to run a single benchmark with the same category profile.
