# EvalScope Agent pipeline acceptance audit

Status: **FINAL CANDIDATE (native local/remote paths and server-side SWE-bench execution verified; quality gate not met)**.

This audit records the fixed experiment configuration, the post-audit
vllm-rwkv interface review, a separate local native-tool run, a remote GPU1
native 2.9B run, and a forwarded 19329 external-bridge comparison. The
EvalScope implementation owns dataset preparation, context assembly, model
requests, raw-response retention, answer extraction, discrimination, and
reporting. vllm-rwkv parser/template changes and model/Agent behavior are
external dependencies; no inference-engine patch is shipped by this branch.
The formal local runner uses the local 19316 service; the remote 19331 service
was a separate temporary native validation endpoint and the existing 19329
process was not modified:

- API: `http://127.0.0.1:19316/v1`
- Model: `rwkv7-g1h-1.5b-20260710-ctx10240`
- API key: `rwkv-skills` (not stored in artifacts)
- Context: `10240` tokens
- Runtime: WSL `uv`, EvalScope `1.9.1`
- Generation: `max_tokens=1024`, `temperature=0.0`

## Requirement matrix

| Requirement | Status | Evidence |
| --- | --- | --- |
| Local RWKV endpoint can be called | PASS | `baseline/chat-naive-preflight.json`; real v3 HTTP runs and EvalScope reports |
| Naive Chat serialization is correct | PASS | `src/cli/helicopter_cli/naive_chat.py`, `src/cli/helicopter_cli/naive_chat_proxy.py`, proxy/serializer tests in `tests/test_naive_chat.py`, `baseline/chat-naive-preflight.json` |
| Existing system and business messages are preserved | PASS | serializer tests cover role/order/content; proxy only transforms the outbound request boundary |
| vllm-rwkv native tool path is configured | EXTERNAL DEPENDENCY | The endpoint must expose OpenAI-compatible `message.tool_calls`; the pipeline only records the response and classifies a missing/malformed call. The temporary parser experiment is not part of this repository or runtime |
| EvalScope Agent datasets are wired | PASS | `benchmarks/evalscope_agent_datasets.json`, CLI catalog listing returns 30 datasets |
| Reproducible benchmark execution | CONDITIONAL | fixed config, local native runs, and server-side SWE-bench one-sample execution complete with `uv`; GAIA remains separately image-gated and formal quality thresholds are not met |
| Raw response and request trace are retained | PASS | v3 `raw/naive_chat.jsonl`, `raw/trace_report.json`, timestamped `raw/acceptance_report.json` |
| Answer extraction is strict and non-repairing | PASS | `evalscope_agent_results.py`; 24 targeted regression tests |
| Discrimination separates transport/format/extraction/model failures | PASS | `tests/test_evalscope_agent_results.py`; live reports show `format_invalid`, `context_truncated`, and official scores separately |
| All discovered issues have regression coverage | PASS | naive Chat, native tool-call validation, direct short answers, timestamped workdir linkage, and acceptance report tests |
| EvalScope Agent regression suite | PASS | `tests/test_naive_chat.py`, `tests/test_evalscope_agent.py`, and `tests/test_evalscope_agent_results.py`: 24 passed |
| Full repository regression | CONDITIONAL | current `uv run --all-groups --with pytest pytest -q`: 302 passed, 23 failures, 6 skipped, 4 warnings, and 406 subtests; failures are outside the EvalScope Agent tests and remain in LightEval/Famous120/benchmark compatibility and sampling-policy areas |
| No blocking pipeline error | PASS (local path); PASS (forwarded external FC path) | local 19316 health check, native preflight, no-proxy EvalScope run, and forwarded 19329 external `general_fc` run all complete; forwarded GAIA is separately blocked by Docker image pull |
| Server-side SWE-bench environment | PASS (pipeline path) | `/home/rwkv/chase/EvalScope`, `swebench==4.1.0`, exact Astropy image, GPU1/19331, container creation/cleanup, scoring and HTML report all completed; inference-engine parser behavior remains external |
| Key benchmark metrics meet the project threshold | NOT MET | local v3 `general_fc tool_call_f1=0`, local GAIA `mean_acc=0`, forwarded 19329 `general_fc tool_call_f1=0`, and native remote GPU1 five-sample `general_fc tool_call_f1=0`; the five-sample run contains positive/negative model decision errors, while controlled one-sample true-negative rounds are correctly classified `correct_no_tool_call` |
| Code, logs, reports, and checkpoints uploaded | PASS | branch `updata/supported-dataset` contains the integration, native/local, forwarded comparison, reproducibility, and server-side SWE-bench evidence commits; tags `evalscope/native-local-code`, `evalscope/native-local-benchmark`, `evalscope/native-local-gaia`, `evalscope/forwarded-external-code`, and `evalscope/reproducible-runner` provide rollback points |
| Every major phase has a rollback checkpoint | PASS | `baseline/*` and `evalscope/*` annotated tags |
| Input-to-report pipeline is automatic | PASS | `helicopter eval evalscope ...` produces EvalScope reports, predictions, reviews, traces, and `acceptance_report.json`; `--report-only` rebuilds evidence |

## Live verification

- `results/evalscope/live-general-fc-v3/20260726_223915/`: one sample,
  official tool-call metrics all zero, raw response ended at the generation cap.
- `results/evalscope/live-gaia-v3/20260726_223929/`: one GAIA level-1 sample,
  two AgentLoop requests, target `17`, official `acc=0`, and no native tool call.
- `post-audit/naive-chat-20260726.json`: latest direct naive-Chat probe returned
  HTTP 200 with prose and no `function_call`; `post-audit/native-tools-20260726.json`
  returned the unchanged HTTP 400 parser configuration error.
- `post-audit/vllm-rwkv-native-tools-20260727.json`: source review and remote
  process inspection show that `rwkv_tool_parser.py` expects the RWKV native
  fenced-JSON protocol, while the running 19329 command omitted
  `--enable-auto-tool-choice` and `--tool-call-parser rwkv`.
- `post-audit/local-1p5b-native-tools-20260727.json`: local 19316 preflight
  returned an OpenAI `tool_calls` object with `finish_reason=tool_calls`.
- `results/evalscope/local-general-fc-1p5b-20260727/20260727_065200/`: one
  no-proxy native EvalScope sample completed with official
  `tool_call_f1=0`; the sample asked for no tool, while the model still
  returned prose. The corrected acceptance report classifies this true negative
  as `correct_no_tool_call` and preserves the raw response.
- `results/evalscope/local-general-fc-1p5b-20260727-limit5/20260727_070506/`:
  five-sample local no-proxy rerun completed with exit code 0; all five raw
  responses are preserved, including one sample with
  `should_call_tool=true`. The corrected diagnostic count is four
  `correct_no_tool_call` true negatives and one `model_error` false negative;
  the official `tool_call_f1=0` remains unchanged.
- `results/evalscope/local-gaia-1p5b-20260727/20260727_071448/`: local native
  GAIA run completed for one sample in each of the three levels with exit code
  0; all three official scores are 0, and the diagnostic report records
  `extraction_failed` for multiline model output rather than silently
  truncating it.
- `post-audit/forwarded-2p9b-native-tools-20260727.json`: forwarded 19329
  returned HTTP 400 for `tool_choice=auto`, identifying missing native parser
  server flags.
- `post-audit/forwarded-2p9b-naive-proxy-20260727.json` and JSONL trace:
  forwarded 2.9B naive Chat request returned HTTP 200 with unchanged prose.
- `post-audit/forwarded-2p9b-native-tools-gpu1-20260727.json`: a separate
  remote GPU1/19331 2.9B service started with
  `--enable-auto-tool-choice --tool-call-parser rwkv` returned HTTP 200 with
  `finish_reason=tool_calls` and a structured `calculate_triangle_area` call.
- `results/evalscope/forwarded-general-fc-2p9b-20260727/20260727_073359/`:
  forwarded 2.9B external bridge + naive proxy completed `general_fc` end to
  end with exit code 0, official tool-call metrics zero, and preserved raw
  trace/acceptance/report artifacts.
- `results/evalscope/forwarded-gaia-2p9b-20260727/20260727_073310/`:
  external bridge task construction completed, but Docker could not pull
  `python:3.11`; no model score is claimed.
- `results/evalscope/forwarded-native-general-fc-2p9b-gpu1-20260727/20260727_081019/`:
  native EvalScope `general_fc` completed on the remote GPU1 2.9B endpoint
  with `tool_call_f1=0`, `count_finish_reason_tool_call=1`,
  `count_successful_tool_call=1`, `schema_accuracy=1`, mean latency 6.6388s,
  and output throughput 148.94 tok/s. The raw predictions show four model
  failures and one valid native tool call; sample 2 required a call but did not
  emit one, while sample 4 emitted a valid call despite `should_call_tool=false`.
  No client-side repair was used.
- `results/evalscope/forwarded-native-gaia-2p9b-gpu1-20260727-function-calling/`:
  corrected native GAIA attempt. EvalScope entered the AgentLoop, but Docker
  could not pull `python:3.11`; the run is not scored.
- `results/evalscope/forwarded-native-general-fc-2p9b-gpu1-20260727-round6/20260727_100621/`:
  one-sample GPU1 rerun with `max_tokens=2048`; metadata says no tool call was
  expected, and the corrected acceptance report records `correct_no_tool_call`.
- `results/evalscope/forwarded-native-general-fc-2p9b-gpu1-20260727-round7-max4096/20260727_100812/`:
  identical sample and endpoint with `max_tokens=4096`; the response ended
  after about 390 tokens without a native tool call and is also
  `correct_no_tool_call`; the one-sample F1 remains 0 because there is no
  positive label. This controlled comparison rules out output truncation as
  the sole cause of the five-sample quality failure.
- `results/evalscope/forwarded-native-swebench-2p9b-gpu1-20260727-server/20260727_092121/`:
  first server-side SWE-bench run after adding the reproducible `swebench`
  dependency group. The exact Astropy image ran, but the model's invalid JSON
  action led to a strict patch-apply failure.
- `results/evalscope/forwarded-native-swebench-2p9b-gpu1-20260727-server-round4/20260727_093947/`:
  patched-parser and `max_tokens=2048` rerun. Native `bash` calls were parsed
  and executed in the Docker sandbox; the official score is 0 because the
  model later requested unavailable `view`/`str_replace_editor` tools and did
  not submit a patch; the corrected diagnostic is `agent_incomplete`, not an
  extraction failure. No client-side answer or patch repair was used.
- `results/evalscope/forwarded-native-swebench-2p9b-gpu1-20260727-server-round8-max4096/20260727_101325/`:
  same Astropy sample with `max_tokens=4096`; 34 native `bash` calls ran, the
  container was cleaned up, and the report was generated. The model repeatedly
  requested unavailable `view`, hit `max_steps_exceeded`, and produced no
  patch, so the corrected diagnostic is `agent_incomplete` and `mean_acc=0`
  remains a model/agent-quality result.
- The CLI catalog lists 30 pinned EvalScope Agent datasets. Native dry-runs for
  `general_fc`, `gaia`, and `swe_bench_verified_agentic` all produced the
  expected EvalScope command with the configured model, endpoint, dataset,
  strategy, and generation settings without contacting a model.
- `experiments/evalscope_agent/post-audit/vllm-rwkv-tool-call-upstream-issue.md`:
  external issue draft describing the observed non-streaming and streaming
  `<tool_calls>` contract. No parser patch is shipped by this repository.
- `uv lock --check`: passed.
- `uv run --no-default-groups --group agent --group eval --with pytest pytest -q tests/test_naive_chat.py tests/test_evalscope_agent.py tests/test_evalscope_agent_results.py`: **24 passed**.
- The answer-extraction regression includes an explicit fullwidth-colon marker
  (`Exact Answer： ...`); direct extraction and the 24-test suite pass.
- Server-side `uv sync --no-default-groups --group agent --group eval --group swe-bench`: completed; `swebench==4.1.0` installed.
- Server-side SWE-bench round 4: exit code 0, 1 sample, 5 model requests, native tool calls executed, `mean_acc=0`, full HTML/JSON/trace artifacts.
- GPU1 `general_fc` rounds 6 and 7: both exit code 0; round 7 used
  `max_tokens=4096`, kept the raw response unchanged, and still recorded
  `correct_no_tool_call` with official `tool_call_f1=0` on the one-sample
  true-negative input.
- GPU1 SWE-bench round 8: exit code 0, `max_tokens=4096`, 5-step loop,
  34 native tool calls, `max_steps_exceeded`, `mean_acc=0`, and complete
  prediction/review/report/acceptance artifacts.

## Go/no-go decision

The software integration and local native endpoint are runnable and
evidence-preserving. The remote GPU1 experiment additionally verifies the
native 2.9B tool-call transport, the server-side `<tool_calls>` parser path,
and the SWE-bench container pipeline required by the requested model
interface. This version must not be labeled a formal passing Agent benchmark
release because the fixed native `general_fc`, SWE-bench, and GAIA scores are
not passing and GAIA still cannot pull its required image. The current adapter
intentionally does not fabricate calls, append answers, or repair truncated
output.

## Latest report-path audit

The `--report-only` path is now portable: absolute workspace paths are
serialized as checkout-relative POSIX paths, so regenerating a saved report
does not embed the local machine's checkout directory. The saved GPU1
`general_fc` acceptance report was regenerated with an absolute work directory
and had no content diff. The focused `uv` regression suite is **52 passed**;
this is an engineering fix and does not change the no-go quality decision above.

## Live endpoint evidence

The current fixed probe confirms the endpoint boundary: 19329 rejects the
tool-call request because its running server lacks the native parser flags,
while the existing GPU1/19331 forward returns a native OpenAI tool call. Both
raw request/response records are saved under
`experiments/evalscope_agent/post-audit/`. The repository does not modify the
inference engine or repair the 19329 response.

The report-only path also preserves the saved run status: it reads
`raw/trace_report.json` instead of assuming exit code zero, and emits `null`
when no trustworthy prior status exists. This prevents report regeneration
from masking an earlier transport or runtime failure; the regression is
covered by the current **53-test** focused `uv` suite.

## FC context audit and current candidate-route gate

The `rwkv-skills` FC implementation is now recorded in
`RWKV_FC_CONTEXT.md`. The local candidate adapter follows its role-labelled
naive Chat serialization, `Assistant: ```json` suffix, newest-history budget,
and lexical long-document evidence window. The adapter retains source
messages and raw completions and does not extract an embedded JSON object from
prose or prune unknown semantic arguments.

The post-audit 13.3B SWE-bench run completed the EvalScope/Docker/reporting
path, but the model emitted prose before its fenced JSON object in all three
decision requests. The strict parser therefore recorded zero valid candidates
and executed no fabricated tool call. This is a model output-contract failure;
the formal quality gate remains **not met**.
