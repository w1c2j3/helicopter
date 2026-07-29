# EvalScope Agent integration experiment record

All model-backed commands in this record were run inside WSL with the
repository environment and `uv run --no-default-groups --no-sync`. Historical
baseline runs and the forwarded comparison use `http://127.0.0.1:19329/v1`
and the SSH-forwarded `rwkv7-g1h-2.9b-20260710-ctx10240` service. The
post-audit native run uses the local-only `http://127.0.0.1:19316/v1` service
with `rwkv7-g1h-1.5b-20260710-ctx10240`. The paths remain separately recorded.
All runs use API key `rwkv-skills` and a 10240-token context limit.

Scope boundary: this repository prepares benchmark data, assembles the
existing messages, requests the model, retains the raw response, extracts an
answer, and discriminates the result. vllm-rwkv parsing/template changes,
tool availability, sandbox execution, and model behavior are external. The
upstream issue note mentioned below is an external contract record only; no
engine patch is shipped, imported, or required by the EvalScope pipeline.

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
  the local runner and the initial 22-test acceptance count; the current
  Agent-specific suite is 24 tests.
- `ed4efd1` / `evalscope/extractor-fullwidth-marker`: the strict answer
  extractor now recognizes the Unicode fullwidth colon in explicit Final
  Answer/Exact Answer markers; the regression suite covers the observed case.
- `64caa2f` / `evalscope/tool-call-contract-diagnostics`: malformed non-list
  `message.tool_calls` values remain `format_invalid`, and an official
  GeneralFC pass cannot override a transport/format/extraction failure.
- `09094e9` / `evalscope/tool-argument-validation`: function arguments are
  accepted only as JSON objects (native object form or valid JSON string),
  while malformed arguments remain explicit `format_invalid` failures.
- The pending server-side SWE-bench checkpoint records the exact remote image
  and the round-4 native run before the GPU was released for reassignment;
  parser behavior is recorded only as an external dependency.
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
- `post-audit/vllm-rwkv-tool-call-upstream-issue.md` records the external issue
  draft for normalizing the checkpoint's `<tool_calls>` JSON-array form to
  OpenAI `message.tool_calls`, including streaming deltas. No engine patch is
  part of this repository's implementation.
- `post-audit/forwarded-2p9b-native-tools-gpu1-round3.screen.log` records the
  restarted GPU1/19331 service during the external parser experiment; that
  engine-side change was not part of this repository and has been removed from
  the deliverable.
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
| `results/evalscope/live-general-fc-v2` | `general_fc`, 1 | `max_tokens=1024`, `temperature=0` | `tool_call_f1=0`, schema/tool-call counts 0 | corrected acceptance report classifies the no-call sample as `correct_no_tool_call` and retains the separate context-limit trace |
| `results/evalscope/live-gaia-v2` | `gaia/2023_level1`, 1 | `max_tokens=1024`, `max_steps=2` | `mean_acc=0` | `acceptance_report.json` joins target `17`, official `acc=0`, two raw requests, and no-tool-call/truncation diagnostics |
| `results/evalscope/live-general-fc-v3/<timestamp>` | `general_fc`, 1 | `max_tokens=1024`, `temperature=0` | `tool_call_f1=0`, schema/tool-call counts 0 | post-extractor rerun; timestamped workdir, official report, predictions/reviews, raw proxy trace, and acceptance report all present |
| `results/evalscope/live-gaia-v3/<timestamp>` | `gaia/2023_level1`, 1 | `max_tokens=1024`, `max_steps=2` | `mean_acc=0` | post-extractor rerun; target `17`, official `acc=0`, two requests, and timestamped acceptance/trace reports all present |
| `results/evalscope/local-general-fc-1p5b-20260727/20260727_065200` | `general_fc`, 1 | local 19316, native tool path, no proxy, `max_steps=3` | `tool_call_f1=0`, schema/tool-call counts 0 | full local prediction/review/report/acceptance artifacts; selected sample had `should_call_tool=false`, and the model returned prose without `tool_calls` |
| `results/evalscope/local-general-fc-1p5b-20260727-limit5/20260727_070506` | `general_fc`, 5 | local 19316, native tool path, no proxy, `max_tokens=1024` | `tool_call_f1=0`, `count_finish_reason_tool_call=0`, `schema_accuracy=0` | all 5 responses retained; corrected diagnostics are four `correct_no_tool_call` true negatives and one `model_error` false negative for the required call; mean latency 4.0416s, output throughput 230.65 tok/s |
| `results/evalscope/local-gaia-1p5b-20260727/20260727_071448` | `gaia`, 3 (one per level) | local 19316, native AgentLoop, no proxy, `max_tokens=1024` | official `mean_acc=0` | full GAIA prediction/review/report/acceptance artifacts; all 3 samples were classified `extraction_failed` because the model returned multiline reasoning where GAIA requires a single-line final answer |
| `results/evalscope/forwarded-general-fc-2p9b-20260727/20260727_073359` | `general_fc`, 1 | forwarded 19329 2.9B, external mock bridge + naive proxy, `max_tokens=1024`, `temperature=0` | official `tool_call_f1=0`, schema/tool-call counts 0 | exit code 0; corrected diagnostic status is `correct_no_tool_call`; mean latency 6.8861s, output throughput 148.71 tok/s |
| `results/evalscope/forwarded-gaia-2p9b-20260727/20260727_073310` | `gaia`, 3 (one per level) | forwarded 19329 2.9B, external mock bridge + naive proxy | not scored | EvalScope entered the external bridge, but all samples were blocked before model calls by Docker failing to pull `python:3.11`; task config and acceptance report are retained |
| `results/evalscope/forwarded-native-general-fc-2p9b-gpu1-20260727/20260727_081019` | `general_fc`, 5 | remote GPU1/19331 2.9B, native vllm-rwkv tool parser, no proxy, `max_tokens=1024`, `temperature=0`, `max_steps=3` | `tool_call_f1=0`, `count_finish_reason_tool_call=1`, `count_successful_tool_call=1`, `schema_accuracy=1` | exit code 0; all five raw predictions retained; mean latency 6.6388s, output throughput 148.94 tok/s. The zero F1 is a model decision/answer-quality failure, not a transport or extraction failure |
| `results/evalscope/forwarded-native-general-fc-2p9b-gpu1-20260727-round6/20260727_100621` | `general_fc`, 1 | remote GPU1/19331 2.9B, native parser, no proxy, `max_tokens=2048`, `temperature=0` | `tool_call_f1=0`, `schema_accuracy=0` | exit code 0; the sample metadata required no tool and the strict diagnostic is now `correct_no_tool_call`; the single true-negative sample has F1 0 because it contains no positive tool-call label |
| `results/evalscope/forwarded-native-general-fc-2p9b-gpu1-20260727-round7-max4096/20260727_100812` | `general_fc`, 1 | same endpoint/messages as round 6, `max_tokens=4096`, `temperature=0` | `tool_call_f1=0`, `schema_accuracy=0` | exit code 0; output stopped at about 390 tokens without a native call and is correctly classified `correct_no_tool_call`; increasing the output cap did not change the model decision |
| `results/evalscope/forwarded-native-gaia-2p9b-gpu1-20260727-function-calling/20260727_081217` | `gaia`, 3 (one per level) | remote GPU1/19331 2.9B, native AgentLoop, `function_calling`, `max_steps=3` | not scored | the corrected run entered the AgentLoop but Docker timed out pulling `python:3.11`, then the temporary 19331 service received SIGTERM. No GAIA score is claimed |
| `results/evalscope/forwarded-native-swebench-2p9b-gpu1-20260727-server/20260727_092121` | `swe_bench_verified_agentic`, 1 | server `/home/rwkv/chase/EvalScope`, remote Docker image, native 19331, `max_tokens=1024`, `max_steps=5` | `mean_acc=0` | full container and EvalScope path completed; raw response used an invalid mini-swe-agent JSON action and patch application failed; no environment blocker |
| `results/evalscope/forwarded-native-swebench-2p9b-gpu1-20260727-server-round4/20260727_093947` | `swe_bench_verified_agentic`, 1 | server `/home/rwkv/chase/EvalScope`, patched RWKV parser, remote Docker image, native 19331, `max_tokens=2048`, `max_steps=5` | `mean_acc=0` | end-to-end pipeline path passed: native `bash` calls executed in the container and raw/review/report/HTML/acceptance artifacts generated; strict diagnostic is `agent_incomplete` because the model requested unavailable tools and did not submit a patch |
| `results/evalscope/forwarded-native-swebench-2p9b-gpu1-20260727-server-round8-max4096/20260727_101325` | `swe_bench_verified_agentic`, 1 | same server/image/parser/tool contract as round 4, `max_tokens=4096`, `max_steps=5` | `mean_acc=0` | exit code 0; 34 native `bash` calls executed, but the model repeatedly requested unavailable `view`, then the loop ended with `max_steps_exceeded` and no patch; strict diagnostic is `agent_incomplete`, sandbox cleanup and full reports passed |

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

The server-side SWE-bench rounds close the environment question. The
`swebench==4.1.0` dependency is now a reproducible `uv` group, the exact
Astropy image was transferred to the server Docker registry, and the server
copy of the project ran EvalScope with the same 2.9B endpoint. The pipeline
received and recorded the native calls and the sandbox container was created
and cleaned up normally. The remaining zero score is a model/tool-contract
failure—after a valid `bash` call the checkpoint requested `view` and
`str_replace_editor`, which this benchmark intentionally does not expose—and
is not converted into a pass.

The round-8 controlled SWE-bench rerun used `max_tokens=4096`. It produced
valid native `bash` calls and container results, but repeatedly issued the
unavailable `view` command and exhausted the five-step loop without a patch.
The increased output budget therefore did not repair the model's tool-contract
behavior; the acceptance report keeps this as a model/agent-quality failure.

The later single-sample GPU1 reruns isolate the generation-cap hypothesis. With
the same input and tool schema, both `max_tokens=2048` and `max_tokens=4096`
returned no tool call for a sample explicitly labelled `should_call_tool=false`.
The strict diagnostic therefore records `correct_no_tool_call` and retains the
raw response; the official F1 remains zero because a one-sample true-negative
run has no true-positive label. The five-sample run remains the authoritative
quality check for positive/negative decision errors.

## Failure classification

- Chat format: fixed for the local endpoint by the naive Chat proxy; verified
  with a real HTTP 200 response.
- Native tool transport: external endpoint contract; verified on local 19316
  with the required server flags, but no vllm-rwkv parser implementation is
  maintained in this repository.
- Forwarded service configuration: 19329 is reachable and serves the expected
  2.9B model, but its live process rejects `tool_choice=auto` without the
  required native parser flags; this is an endpoint configuration issue, not
  an extraction or discriminator failure.
- Interface error: reproduced as the original HTTP 400 tool-call request.
- Model/format failure: a `general_fc` response that omits a required tool call
  or emits a tool call for a `should_call_tool=false` sample is a model decision
  failure; an explicitly expected no-call response is `correct_no_tool_call`.
- Agent tool-contract failure: SWE-bench round 8 emitted valid `bash` calls but
  repeatedly selected the unavailable `view` command and ended at
  `max_steps_exceeded` without a patch; no tool name was rewritten or hidden.
- Agent runtime: the first GAIA run exposed the missing `ms_enclave` extra;
  installing `evalscope[sandbox]` with `uv pip` removed that import blocker for
  the fixed-subset rerun.
- Context truncation: reproduced at the 10240 total-token boundary and then
  retained as an explicit `context_truncated` diagnostic.
- Extraction failure: strict answer extractors return a failure status rather
  than inventing a label, number, code fence, JSON field, or tool call.
- Discriminator error: no known reproduction; regression tests cover transport,
  format, extraction, and strict model-error separation.
- Malformed tool-call contract: a non-list `message.tool_calls` object is
  retained as raw output and classified `format_invalid`; it is never silently
  interpreted as a correct no-call response.
- Malformed tool arguments: invalid JSON and non-object arguments are rejected
  as `format_invalid`; the 46 object-form arguments in the saved EvalScope
  predictions remain unchanged.
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
- RWKV native parser: the direct long-context request produced the model's
  `<tool_calls>` array. If an endpoint does not map that wrapper to OpenAI
  `message.tool_calls`, it is an upstream vllm-rwkv issue. The EvalScope
  pipeline preserves the raw response and reports the contract failure; it
  does not rewrite the response.
- SWE-bench environment: the server-side run uses the existing exact Astropy
  image and has no dependency, Docker, or interface blocker. The sample still
  scores zero because the model did not finish a patch.
- GAIA sandbox: the earlier GAIA run remains environment-blocked when it needs
  `python:3.11`; the SWE-bench image transfer does not claim to fix that
  separate image requirement.

## Regression boundary (2026-07-27)

- Agent-specific suite: **24 passed** with the Agent, naive-Chat, extraction,
  and discrimination tests.
- Full repository suite from the current worktree with
  `uv run --all-groups --with pytest pytest -q`: **302 passed, 23 failed, 6
  skipped, 4 warnings, 406 subtests**. The failures are in LightEval/Famous120,
  benchmark compatibility, and sampling-policy assertions; no EvalScope Agent
  test failed.
- The focused LightEval answer-adapter suite remains **27 passed** and is
  separate from the unrelated full-suite failures.

## Remaining formal-evaluation limits

`swe_bench_verified_agentic` now has a complete server-side one-sample run,
including dependency installation, exact Docker image, native tool parsing,
sandbox execution, scoring, and report generation. Its score is zero because
the model did not finish a patch; this is a quality result, not a blocked run.
The current local 19316 process has an actual native tool-call parser, but the
available local checkpoint is the 1.5B variant. The forwarded 19329 remains
the existing 2.9B service and was not restarted; it lacks native parser flags.
A separate GPU1/19331 instance now proves both the requested 2.9B native
transport and the `<tool_calls>` parser path, but the fixed native
`general_fc` and SWE-bench scores remain zero. GAIA still has its separate
Docker registry limitation. These are recorded limits, not silently converted
into passing results.

## Report portability audit (2026-07-27)

- The acceptance-report regeneration check found that `--report-only` could
  serialize an absolute `--work-dir` into `output_dir`, sample artifact paths,
  and official report paths. This made an otherwise identical report depend on
  the machine checkout path.
- `_report_path` now emits workspace-relative POSIX paths for artifacts below
  the current checkout and preserves an absolute path only when an artifact is
  genuinely outside the workspace. No model output, extraction result, or
  score is changed.
- Re-running `--report-only` against the saved GPU1 `general_fc` run with an
  absolute work directory produced no report diff. The focused regression suite
  is **52 passed**, and `uv lock --check` passed.

## Live endpoint audit (2026-07-27)

- `experiments/evalscope_agent/post-audit/live-tool-probe-19329-20260727.json`
  records a fixed tool-call request to the existing 19329 endpoint. It returned
  HTTP 400 with the server's explicit requirement for
  `--enable-auto-tool-choice` and `--tool-call-parser`; the endpoint was not
  restarted or reconfigured.
- `experiments/evalscope_agent/post-audit/live-tool-probe-19331-20260727.json`
  records the same request through the existing GPU1/19331 forward. It returned
  HTTP 200, `finish_reason=tool_calls`, and a structured
  `calculate_triangle_area` call with valid JSON-object arguments.
- The 19331 response also retains the model's prose and `</think>` text in the
  raw content. The pipeline keeps that content unchanged and extracts only the
  native tool-call field; no answer or tag is added by the client.

## Report-only exit-code audit (2026-07-27)

- A report-only review found that the CLI previously hard-coded
  `exit_code=0`, which could make a failed saved run appear successful.
- Report-only regeneration now reads the exit code from the saved
  `raw/trace_report.json`; if that trace is absent or malformed, the report
  records `null` rather than claiming success.
- The new regression creates a saved trace with exit code 17 and verifies that
  report-only preserves 17. The focused `uv` suite now passes **53 tests**.
## rwkv-skills FC context audit and parallel-candidate rerun

The local `rwkv-skills` FC implementation was audited before changing the
candidate route. Its protocol is role-labelled naive Chat, not a JSON dump of
the OpenAI messages array: the upstream prompt uses `System:`, `User:`,
`Assistant:`, and `User: Function output:` blocks and ends at
`Assistant: ```json`. History is retained newest-first under a character
budget, with an explicit truncation marker. Long messages are lexically
chunked with line overlap and selected against the latest short user query;
the selected chunk IDs and evidence-window metadata are traceable. See
`RWKV_FC_CONTEXT.md` for the exact reference paths and contract.

The local parallel-candidate adapter now follows that contract. The raw
EvalScope request is preserved unchanged; only the outbound model prompt is
role-rendered. It applies deterministic long-document compaction before
history trimming and records both traces. The candidate parser accepts only a
complete leading JSON object (plus an optional closing fence), preserves raw
transport `id` and stringified `arguments` compatibility, and rejects prose
prefixes, trailing repeated objects, unknown tool fields, and missing required
arguments.

Focused regression tests after this change: **29 passed** via the documented
`uv run` command in `RWKV_FC_CONTEXT.md`.

Controlled post-audit run:

| Run | Configuration | Result | Classification |
| --- | --- | --- | --- |
| `results/evalscope/parallel-candidate-13p3b-swebench-20260728-round2/20260728_182039/` | 13.3B at `29534`, `swe_bench_verified_agentic`, one sample, `max_steps=3`, candidate/aggregate caps `2048/1024`, local x86_64 image config | EvalScope, Docker setup/cleanup, reports and trace completed; `mean_acc=0` | 3 model completions were prose-prefixed; strict parser returned `candidate_count=0`, so no tool was executed or fabricated |

This rerun deliberately does not reinterpret an embedded fenced object as a
valid call. The model must emit the required JSON call contract directly; the
raw completion and parser reason remain in `raw/parallel_candidate.jsonl`.

The post-audit `general_fc` controls used the same adapter and fixed
`max_tokens=4096` on both forwarded endpoints. The 7.2B/29572 run is at
`results/evalscope/parallel-candidate-7p2b-general-fc-20260728-post-audit/20260728_182529/`;
the 13.3B/29534 run is at
`results/evalscope/parallel-candidate-13p3b-general-fc-20260728-post-audit/20260728_182529/`.
Both completed with official `tool_call_f1=0`, one strict parser failure, and
no candidate selected. The raw completions are retained; this is a model
output-contract result, not a successful extraction.

## Simple environment-free benchmark

For a benchmark without Docker, browser, MCP, or repository setup, the
recommended smoke test is `general_fc`. The fixed 5-sample run used the 7.2B
endpoint at `http://127.0.0.1:29572/v1`, `temperature=0`, `max_tokens=1024`,
candidate/aggregate caps `1024/512`, and the parallel-candidate route:

`results/evalscope/simple-general-fc-7p2b-20260728/20260728_184744/`

EvalScope completed normally and generated predictions, reviews, reports,
raw proxy traces, and an acceptance report. The official metrics were
`tool_call_f1=0`, `count_finish_reason_tool_call=0`,
`count_successful_tool_call=0`, and `schema_accuracy=0` for 5 samples. The
proxy trace records five strict parser failures (`length` for samples 1, 2,
and 5; `stop` for samples 3 and 4); no candidate was selected and no tool
call was fabricated.
