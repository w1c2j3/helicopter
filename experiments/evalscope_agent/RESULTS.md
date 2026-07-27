# EvalScope Agent integration experiment record

All model-backed commands in this record were run inside WSL with the
repository environment and `uv run --no-default-groups --no-sync`. Historical
baseline runs and the forwarded comparison use `http://127.0.0.1:19329/v1`
and the SSH-forwarded `rwkv7-g1h-2.9b-20260710-ctx10240` service. The
post-audit native run uses the local-only `http://127.0.0.1:19316/v1` service
with `rwkv7-g1h-1.5b-20260710-ctx10240`. The paths remain separately recorded.
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
- `61187fe` / `evalscope/native-local-benchmark`: local native preflight,
  EvalScope reports, raw responses, and the acceptance audit are committed and
  pushed.
- `4cb223f` / `evalscope/native-local-gaia`: local GAIA native AgentLoop
  reports and strict multiline-answer extraction diagnostics are committed and
  pushed.
- `e58c541` / `evalscope/forwarded-external-code`: EvalScope 1.9.x external
  discriminator compatibility, public `--mode external`, and CLI regression
  coverage.
- `41cc021` / forwarded 2.9B experiment: 19329 native preflight, naive proxy
  trace, external `general_fc` run, GAIA sandbox-blocked run, and audit records.
- `ad84822` / `evalscope/reproducible-runner`: uv dependency-group selection in
  the local runner and the corrected 22-test acceptance count.
- `5f7171d`: regression boundary and the distinction between Agent-specific
  passes and unrelated LightEval compatibility failures are recorded.
- The timestamped-workdir fix is validated by the v3 reruns below: the inner
  EvalScope work directory now contains the acceptance report and a copy of
  the raw trace summary, while the outer directory retains the proxy JSONL.
- `post-audit/vllm-rwkv-native-tools-20260727.json` records the source-level
  parser audit and the exact server flags required for native RWKV tool calls.
- `post-audit/local-1p5b-native-tools-20260727.json` records a real local
  HTTP response containing an OpenAI `tool_calls` object and
  `finish_reason=tool_calls`.
- `post-audit/forwarded-2p9b-native-tools-20260727.json` records the forwarded
  2.9B service rejecting an OpenAI tool request because its running server was
  not started with the auto-tool-choice/parser flags.
- `post-audit/forwarded-2p9b-native-tools-gpu1-20260727.json` records a
  separate 2.9B service on remote GPU1/port 19331 started with both native
  RWKV parser flags. It returned HTTP 200, `finish_reason=tool_calls`, and a
  structured `calculate_triangle_area` call; the existing 19329 service was
  not modified.
- `post-audit/forwarded-2p9b-native-tools-gpu1-20260727.screen.log` is the
  captured vLLM startup, request, and normal SIGTERM shutdown log for that
  temporary GPU1 service.
- `post-audit/forwarded-2p9b-naive-proxy-20260727.json` and its JSONL trace
  record a successful 19329 request through the naive Chat proxy. The model
  returned prose and no `tool_calls`; the proxy preserved that response.

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
| `results/evalscope/local-gaia-1p5b-20260727/20260727_071448` | `gaia`, 3 (one per level) | local 19316, native AgentLoop, no proxy, `max_tokens=1024` | official `mean_acc=0` | full GAIA prediction/review/report/acceptance artifacts; all 3 samples were classified `extraction_failed` because the model returned multiline reasoning where GAIA requires a single-line final answer |
| `results/evalscope/forwarded-general-fc-2p9b-20260727/20260727_073359` | `general_fc`, 1 | forwarded 19329 2.9B, external mock bridge + naive proxy, `max_tokens=1024`, `temperature=0` | official `tool_call_f1=0`, schema/tool-call counts 0 | exit code 0; raw response reached the generation cap, diagnostic status `context_truncated` plus `format_invalid`; mean latency 6.8861s, output throughput 148.71 tok/s |
| `results/evalscope/forwarded-gaia-2p9b-20260727/20260727_073310` | `gaia`, 3 (one per level) | forwarded 19329 2.9B, external mock bridge + naive proxy | not scored | EvalScope entered the external bridge, but all samples were blocked before model calls by Docker failing to pull `python:3.11`; task config and acceptance report are retained |
| `results/evalscope/forwarded-native-general-fc-2p9b-gpu1-20260727/20260727_081019` | `general_fc`, 5 | remote GPU1/19331 2.9B, native vllm-rwkv tool parser, no proxy, `max_tokens=1024`, `temperature=0`, `max_steps=3` | `tool_call_f1=0`, `count_finish_reason_tool_call=1`, `count_successful_tool_call=1`, `schema_accuracy=1` | exit code 0; all five raw predictions retained; mean latency 6.6388s, output throughput 148.94 tok/s. The zero F1 is a model decision/answer-quality failure, not a transport or extraction failure |
| `results/evalscope/forwarded-native-gaia-2p9b-gpu1-20260727-function-calling/20260727_081217` | `gaia`, 3 (one per level) | remote GPU1/19331 2.9B, native AgentLoop, `function_calling`, `max_steps=3` | not scored | the corrected run entered the AgentLoop but Docker timed out pulling `python:3.11`, then the temporary 19331 service received SIGTERM. No GAIA score is claimed |

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

The forwarded 2.9B `general_fc` run proves the current `external` bridge
configuration, EvalScope 1.9.1 task construction, model request, proxy trace,
official report, and strict diagnostics end to end. It is a compatibility
comparison, not a passing tool-call result: the remote server's native parser
flags are absent, while the external mock runner sends a text-only request and
the model response ends at the 1024-token cap.

The GPU1 run closes the transport question for the requested 2.9B checkpoint:
the same model, when served independently with
`--enable-auto-tool-choice --tool-call-parser rwkv`, returns native OpenAI
tool calls. EvalScope then sends and receives native calls without the naive
proxy. Its official F1 remains zero on the fixed five-sample benchmark because
the model emits a call on only one sample. The official sample metadata makes
the failure concrete: sample 2 has `should_call_tool=true` but no call was
emitted, while sample 4 has `should_call_tool=false` but the model emitted a
schema-valid `search` call. The successful-call and schema metrics are retained
separately so these model decision errors are not misclassified as an adapter
failure.

## Failure classification

- Chat format: fixed for the local endpoint by the naive Chat proxy; verified
  with a real HTTP 200 response.
- Native tool transport: fixed in the managed server plan and verified on
  local 19316 with `--enable-auto-tool-choice --tool-call-parser rwkv`.
- Forwarded service configuration: 19329 is reachable and serves the expected
  2.9B model, but its live process rejects `tool_choice=auto` without the
  required native parser flags; this is an endpoint configuration issue, not
  an extraction or discriminator failure.
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
- GAIA extraction: reproduced a real multiline final response and correctly
  rejected it as `extraction_failed` rather than truncating reasoning into an
  answer; the raw response and official score remain in the local GAIA report.
- External bridge: the first forwarded run exposed the invalid legacy config
  discriminator (`bridge` emitted without `mode`). The adapter now emits
  EvalScope 1.9.x `mode=external` and keeps `bridge` only as a CLI alias; the
  forwarded `general_fc` rerun completed with exit code 0.
- Remote native 2.9B: a separately launched GPU1/19331 service with the RWKV
  parser flags passed the direct native tool-call preflight and the native
  EvalScope `general_fc` transport path. The official F1 failure is attributed
  to model behavior on the fixed dataset. The service later shut down on an
  external SIGTERM after the GAIA Docker pull timeout; this was not an OOM or
  vLLM parser crash.
- GAIA sandbox: the corrected native run still cannot be scored in this
  environment because Docker cannot pull `python:3.11` from the registry.

## Regression boundary (2026-07-27)

- Agent-specific suite: **22 passed** with the Agent, naive-Chat, extraction,
  and discrimination tests.
- Full repository suite after `uv sync --no-default-groups --group agent
  --group eval`: **292 passed, 6 skipped, 22 failed**. The failures are in
  existing LightEval/Famous120/benchmark compatibility assertions; no
  EvalScope Agent test failed.
- Excluding `tests/test_cli.py` and
  `tests/test_lighteval_answer_adapters.py`: **23 passed, 1 failed**. The one
  remaining failure is the existing `commands_sampling` expectation for a
  LightEval math token budget and is outside the EvalScope Agent path.

## Remaining formal-evaluation limits

`swe_bench_verified_agentic` was not scored because EvalScope 1.9.1 requires
the optional `swebench` package; its extra installation exceeded the bounded
experiment window. The current local 19316 process has an actual native
tool-call parser, but the available local checkpoint is the 1.5B variant. The
forwarded 19329 remains the existing 2.9B service and was not restarted; it
lacks native parser flags. A separate GPU1/19331 instance proved that the
requested 2.9B checkpoint can return native tool calls, but the fixed native
`general_fc` score is still zero. GAIA can enter the AgentLoop, but this run
was blocked by Docker image availability. These are recorded limits, not
silently converted into passing results.
