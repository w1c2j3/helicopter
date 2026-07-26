# EvalScope Agent integration experiment record

All model-backed commands in this record were run inside WSL with the
repository environment and `uv run --no-default-groups --no-sync`. Historical
baseline runs used `http://127.0.0.1:19329/v1` and the SSH-forwarded
`rwkv7-g1h-2.9b-20260710-ctx10240` service. The post-audit native run uses the
local-only `http://127.0.0.1:19316/v1` service with
`rwkv7-g1h-1.5b-20260710-ctx10240`; it does not use the forwarded endpoint.
All runs use API key `rwkv-skills` and a 10240-token context limit.

## Checkpoints

- `4a68cb0` / `baseline/pre-evalscope`: original pre-integration code.
- `24ab52c` / `baseline/wsl-actual`: corrected WSL preflight evidence.
- `adc2500`: naive Chat transport adapter and EvalScope command wiring.
- `52582e5` / `evalscope/strict-diagnostics`: strict extraction, discrimination,
  trace reporting, and the first live run artifacts.
- `bd3c7d6` / `evalscope/agent-only-cli`: LightEval and native FC modules are
  now imported only for their respective subcommands, so dataset listing and
  EvalScope dry-runs do not require the large LightEval dependency group.
- The acceptance-report implementation generates `raw/acceptance_report.json`
  after every EvalScope run and can rebuild it with `uv run
  --no-default-groups --no-sync helicopter eval evalscope --report-only
  --work-dir <run-dir>`.
- `ca1a35d` / `evalscope/direct-answer-contract`: direct one-line answers are
  accepted only for datasets whose output contract is single-line; multiline
  reasoning is never truncated into an answer.
- `df4974e` / `evalscope/native-local-code`: managed vllm-rwkv native parser
  flags, local-only model catalog and runner, native EvalScope defaults, and
  regression coverage are committed and pushed.
- The timestamped-workdir fix is validated by the v3 reruns below: the inner
  EvalScope work directory now contains the acceptance report and a copy of
  the raw trace summary, while the outer directory retains the proxy JSONL.
- `post-audit/vllm-rwkv-native-tools-20260727.json` records the source-level
  parser audit and the exact server flags required for native RWKV tool calls.
- `post-audit/local-1p5b-native-tools-20260727.json` records a real local
  HTTP response containing an OpenAI `tool_calls` object and
  `finish_reason=tool_calls`.

## Baseline and transport probes

The original function-call request reached the local service but returned HTTP
400: the server was not launched with `--enable-auto-tool-choice` and a tool
call parser. The same existing system/user content converted to one naive Chat
message returned HTTP 200. The raw records are in
`baseline/chat-preflight.json` and `baseline/chat-naive-preflight.json`.

The proxy probe returned HTTP 200 and preserved the response unchanged. Its
forwarded request shows the original message order and tool schema metadata in
the transcript; it does not add an answer, tool call, or end marker to the
model response. See `naive-proxy-trace.jsonl`.

## Live EvalScope runs

| Run | Dataset/limit | Generation | Official result | Diagnostic result |
| --- | --- | --- | --- | --- |
| `results/evalscope/live-general-fc` | `general_fc`, 1 | default, total reached 10240 | all scores 0 | raw output ended with `finish_reason=length`; old run had no diagnostic report |
| `results/evalscope/live-general-fc-v2` | `general_fc`, 1 | `max_tokens=1024`, `temperature=0` | `tool_call_f1=0`, schema/tool-call counts 0 | `acceptance_report.json` records missing native `tool_calls` as `format_invalid` and retains the separate context-limit trace |
| `results/evalscope/live-gaia-v2` | `gaia/2023_level1`, 1 | `max_tokens=1024`, `max_steps=2` | `mean_acc=0` | `acceptance_report.json` joins target `17`, official `acc=0`, two raw requests, and no-tool-call/truncation diagnostics |
| `results/evalscope/live-general-fc-v3/<timestamp>` | `general_fc`, 1 | `max_tokens=1024`, `temperature=0` | `tool_call_f1=0`, schema/tool-call counts 0 | post-extractor rerun; timestamped workdir, official report, predictions/reviews, raw proxy trace, and acceptance report all present |
| `results/evalscope/live-gaia-v3/<timestamp>` | `gaia/2023_level1`, 1 | `max_tokens=1024`, `max_steps=2` | `mean_acc=0` | post-extractor rerun; target `17`, official `acc=0`, two requests, and timestamped acceptance/trace reports all present |
| `results/evalscope/local-general-fc-1p5b-20260727/20260727_065200` | `general_fc`, 1 | local 19316, native tool path, no proxy, `max_steps=3` | `tool_call_f1=0`, schema/tool-call counts 0 | full local prediction/review/report/acceptance artifacts; selected sample had `should_call_tool=false`, and the model returned prose without `tool_calls` |
| `results/evalscope/local-general-fc-1p5b-20260727-limit5/20260727_070506` | `general_fc`, 5 | local 19316, native tool path, no proxy, `max_tokens=1024` | `tool_call_f1=0`, `count_finish_reason_tool_call=0`, `schema_accuracy=0` | all 5 responses retained; sample 2 required a tool, but all five were classified `format_invalid` because the model returned text without `tool_calls`; mean latency 4.0416s, output throughput 230.65 tok/s |

The second run proves the complete path: ModelScope dataset loading, proxy
serialization, local model call, unchanged response, EvalScope report, and
Helicopter `raw/trace_report.json`. The local native run proves the separate
native path: the vllm-rwkv server emits an OpenAI `tool_calls` object without
client-side repair. Its benchmark score remains zero because the selected
sample did not require a tool and the model response was prose; the acceptance
report keeps this as a format/model outcome rather than treating it as a
successful tool call.

The GAIA run additionally proved EvalScope's native AgentLoop path and was
rerun after installing `evalscope[sandbox]` with `uv pip`. Its fixed subset is
recorded in `gaia-level1.json`. The model repeatedly described the intended
bash action instead of emitting a tool call and reached the 1024-token cap;
the official score is therefore zero and the raw two-request trace is kept.

## Failure classification

- Chat format: fixed for the local endpoint by the naive Chat proxy; verified
  with a real HTTP 200 response.
- Native tool transport: fixed in the managed server plan and verified on
  local 19316 with `--enable-auto-tool-choice --tool-call-parser rwkv`.
- Interface error: reproduced as the original HTTP 400 tool-call request.
- Model/format failure: `general_fc` response did not contain a tool call.
- Agent runtime: the first GAIA run exposed the missing `ms_enclave` extra;
  installing `evalscope[sandbox]` with `uv pip` removed that import blocker for
  the fixed-subset rerun.
- Context truncation: reproduced at the 10240 total-token boundary and then
  retained as an explicit `context_truncated` diagnostic.
- Extraction failure: strict answer extractors return a failure status rather
  than inventing a label, number, code fence, JSON field, or tool call.
- Discriminator error: no known reproduction; regression tests cover transport,
  format, extraction, and strict model-error separation.
- Clean Agent environment: after `uv sync --no-default-groups --group agent`,
  dataset listing and EvalScope dry-run now pass without importing LightEval;
  the unrelated legacy `tests/test_cli.py` still needs the full LightEval group.

## Remaining formal-evaluation limits

`swe_bench_verified_agentic` was not scored because EvalScope 1.9.1 requires
the optional `swebench` package; its extra installation exceeded the bounded
experiment window. The current local 19316 process has an actual native
tool-call parser, but the available local checkpoint is the 1.5B variant; the
historical 19329 tunnel remains a baseline-only endpoint and was not used for
the native run. GAIA can now enter the AgentLoop after the sandbox extra is
installed, but the model still fails its required tool-call format on the fixed
sample. These are recorded limits, not silently converted into passing results.
