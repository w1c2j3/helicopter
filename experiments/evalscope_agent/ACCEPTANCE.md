# EvalScope Agent pipeline acceptance audit

Status: **FINAL CANDIDATE — native local tool path verified; quality gate not met**.

This audit records the fixed experiment configuration, the post-audit
vllm-rwkv source review, and a separate local native-tool run. The historical
19329 endpoint is retained only as baseline evidence; the current native run
uses the local 19316 service and never uses the SSH-forwarded model:

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
| vllm-rwkv native tool path is configured | PASS (local service) | `vllm-rwkv` `rwkv_tool_parser.py`, native RWKV chat template, `scripts/start_local_rwkv_tool_server.sh`, and `post-audit/vllm-rwkv-native-tools-20260727.json`; local 19316 starts with both required flags |
| EvalScope Agent datasets are wired | PASS | `benchmarks/evalscope_agent_datasets.json`, CLI catalog listing returns 30 datasets |
| Reproducible benchmark execution | CONDITIONAL | fixed config and local native `general_fc` and GAIA runs complete; SWE-bench and other official harnesses remain dependency- and environment-gated |
| Raw response and request trace are retained | PASS | v3 `raw/naive_chat.jsonl`, `raw/trace_report.json`, timestamped `raw/acceptance_report.json` |
| Answer extraction is strict and non-repairing | PASS | `evalscope_agent_results.py`; 21 targeted regression tests |
| Discrimination separates transport/format/extraction/model failures | PASS | `tests/test_evalscope_agent_results.py`; live reports show `format_invalid`, `context_truncated`, and official scores separately |
| All discovered issues have regression coverage | PASS | naive Chat, native tool-call validation, direct short answers, timestamped workdir linkage, and acceptance report tests |
| EvalScope Agent regression suite | PASS | `tests/test_naive_chat.py`, `tests/test_evalscope_agent.py`, and `tests/test_evalscope_agent_results.py`: 21 passed |
| Full repository regression | CONDITIONAL | full run: 292 passed, 6 skipped, 22 failures in existing LightEval/Famous120/benchmark compatibility tests; no EvalScope Agent test failed. Excluding the two LightEval test files leaves 23 passed and one pre-existing sampling-budget failure |
| No blocking pipeline error | PASS (local path) | local 19316 health check, native preflight, and no-proxy EvalScope run all complete |
| Key benchmark metrics meet the project threshold | NOT MET | v3 `general_fc tool_call_f1=0` and GAIA `mean_acc=0`; the model emitted prose and no native `tool_calls` |
| Code, logs, reports, and checkpoints uploaded | PASS | branch `updata/supported-dataset` contains commits `df4974e`, `61187fe`, and `4cb223f`; tags `evalscope/native-local-code`, `evalscope/native-local-benchmark`, and `evalscope/native-local-gaia` provide rollback points |
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
  returned prose. The acceptance report classifies the missing tool object as
  `format_invalid` and preserves the raw response.
- `results/evalscope/local-general-fc-1p5b-20260727-limit5/20260727_070506/`:
  five-sample local no-proxy rerun completed with exit code 0; all five raw
  responses are preserved, including one sample with
  `should_call_tool=true`. The model emitted no native tool call in any sample,
  so the diagnostic count is `format_invalid=5` and the official
  `tool_call_f1=0`.
- `results/evalscope/local-gaia-1p5b-20260727/20260727_071448/`: local native
  GAIA run completed for one sample in each of the three levels with exit code
  0; all three official scores are 0, and the diagnostic report records
  `extraction_failed` for multiline model output rather than silently
  truncating it.
- `uv lock --check`: passed.
- `uv run --no-default-groups --no-sync pytest -q tests/test_naive_chat.py tests/test_evalscope_agent.py tests/test_evalscope_agent_results.py`: **21 passed**.

## Go/no-go decision

The software integration and local native endpoint are runnable and
evidence-preserving. This version must not be labeled a formal passing Agent
benchmark release because the fixed local model run scored zero and the model
did not emit a tool call for the selected benchmark sample. The current adapter
intentionally does not fabricate calls, append answers, or repair truncated
output.
