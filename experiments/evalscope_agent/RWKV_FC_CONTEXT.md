# RWKV FC context and naive Chat contract

This note records the local audit of `rwkv-skills` used for the EvalScope
Agent adapter. The reference files are:

- `src/eval/tasks/function_calling/context_budget.py`
- `src/eval/tasks/function_calling/rwkv_prompt.py`
- `src/eval/experiments/parallel_candidate_router/router.py`
- `src/eval/long_doc_evidence.py`
- `src/plugins/lexical_chunk_router/long_doc.py`

## Message serialization

The reference route does not JSON-encode the whole OpenAI `messages` array.
It builds one naive Chat transcript:

```text
System: <router instructions>

System: <original system content, unchanged>

User: <user content>

Assistant: <previous natural-language assistant content>

User: Function output:
<tool result>

Assistant: ```json
```

The ordinary FC prompt may prefill the final `{`; the parallel-candidate
router uses `assistant_json_prefix(prefill_object=False)`, so its completion
must begin with a complete JSON object after the ` ```json` marker. Transport
metadata such as `id` and a stringified JSON `arguments` field is normalized
only as a transport detail. Tool names, required arguments, and argument
property names remain schema-validated.

The helicopter adapter keeps the original EvalScope request in the raw trace,
renders role-labelled messages only for the upstream model request, and
returns a tool call only after strict validation. It does not select an object
from prose, append a missing brace, prune unknown semantic arguments, or
invent a tool call.

## History budget

The reference history budget is character-based. Messages are considered from
newest to oldest. The newest complete messages are retained; if an older
message must be cut, its tail is retained when there is room and
`[Earlier conversation history truncated]` is inserted. This preserves recent
tool state without silently pretending that the complete history was present.

The adapter exposes the same controls through `--candidate-context-chars` and
`--candidate-prompt-max-chars`. The prompt builder first applies the history
budget and then reduces the history budget again if the complete prompt still
exceeds the hard prompt budget. Prompt and truncation metadata are saved in
`raw/parallel_candidate.jsonl`.

## Long-document budget

For a message at or above `long_doc_min_chars` (default 6000), the reference
implementation performs deterministic lexical compaction:

1. split on newline boundaries, splitting overlong lines and retaining a small
   line overlap;
2. infer the query from the latest short user message;
3. score chunks by query-term presence;
4. keep the highest-scoring chunks subject to chunk-count and evidence-character
   limits; and
5. replace the long message with an explicit evidence window containing the
   original character count, chunk count, selected chunk IDs, line ranges, and
   scores.

The helicopter adapter implements the same evaluation-layer operation in
`rwkv_agent_prompt.py`. It is applied before the history budget, and its
`selected_messages` trace is retained. No model-generated summary is inserted
and an unselected document is represented explicitly as `[No evidence chunk
selected.]`.

## Verification

The focused regression suite covers role/order preservation, function-output
rendering, the ` ```json` suffix, newest-history retention, explicit
truncation, long-document chunk metadata, stringified transport arguments,
and rejection of prose-prefixed or trailing multi-object completions.

The post-audit 13.3B SWE-bench run at
`results/evalscope/parallel-candidate-13p3b-swebench-20260728-round2/20260728_182039`
completed the EvalScope and Docker pipeline. The model returned prose followed
by a fenced JSON call, not a response beginning with the required JSON object;
the strict adapter recorded `candidate_count=0` and did not execute a tool.
This is retained as a model output-contract failure, not hidden by extracting
the first embedded object.
