# Upstream issue draft: RWKV tool-call response normalization

Suggested title:

`OpenAI-compatible tool-call parser should normalize <tool_calls> array output, including streaming deltas`

Observed behavior:

- The RWKV checkpoint can emit a `<tool_calls>...</tool_calls>` wrapper whose
  payload is a JSON array of function calls.
- An OpenAI-compatible server should expose that payload as
  `choices[0].message.tool_calls` and set the corresponding tool-call finish
  reason.
- The same normalization is required for streaming deltas.

Expected behavior:

1. Parse the wrapper only at the inference-engine boundary.
2. Preserve function names and JSON arguments without adding or repairing
   model content.
3. Reject malformed payloads explicitly rather than returning a prose response
   that looks like a successful no-call result.
4. Keep non-tool responses unchanged.

Scope note: this is an upstream vllm-rwkv issue. The Helicopter EvalScope
pipeline does not modify the parser, rewrite model output, or provide an
engine-side compatibility shim. It records the raw response and classifies a
missing or malformed OpenAI tool-call object as an observable contract failure.
