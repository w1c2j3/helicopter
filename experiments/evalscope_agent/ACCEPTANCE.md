# EvalScope Agent pipeline acceptance audit

Status: **FINAL CANDIDATE — not a formal high-quality release**.

This audit is tied to commit `04c5c9d` and the fixed experiment configuration:

- API: `http://127.0.0.1:19329/v1`
- Model: `rwkv7-g1h-2.9b-20260710-ctx10240`
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
| EvalScope Agent datasets are wired | PASS | `benchmarks/evalscope_agent_datasets.json`, CLI catalog listing returns 30 datasets |
| Reproducible benchmark execution | CONDITIONAL | fixed config and real `general_fc`/GAIA v3 runs pass; SWE-bench was not scored because its optional verifier dependency is separate |
| Raw response and request trace are retained | PASS | v3 `raw/naive_chat.jsonl`, `raw/trace_report.json`, timestamped `raw/acceptance_report.json` |
| Answer extraction is strict and non-repairing | PASS | `evalscope_agent_results.py`; 17 targeted regression tests |
| Discrimination separates transport/format/extraction/model failures | PASS | `tests/test_evalscope_agent_results.py`; live reports show `format_invalid`, `context_truncated`, and official scores separately |
| All discovered issues have regression coverage | PASS | naive Chat, native tool-call validation, direct short answers, timestamped workdir linkage, and acceptance report tests |
| No blocking pipeline error | CONDITIONAL | pipeline completes end-to-end; model capability blocker remains for tool-use scoring |
| Key benchmark metrics meet the project threshold | NOT MET | v3 `general_fc tool_call_f1=0` and GAIA `mean_acc=0`; the model emitted prose and no native `tool_calls` |
| Code, logs, reports, and checkpoints uploaded | CONDITIONAL | annotated EvalScope tags are uploaded; exact remote branch push is rejected by the existing top-level `chase` ref conflict |
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
- `uv lock --check`: passed.
- `uv run --no-default-groups --no-sync pytest -q tests/test_naive_chat.py tests/test_evalscope_agent.py tests/test_evalscope_agent_results.py`: **17 passed**.

## Go/no-go decision

The software integration is runnable and evidence-preserving, but this version
must not be labeled a formal passing Agent benchmark release until the endpoint
either emits valid OpenAI `tool_calls` or the model/server configuration changes
externally. The current adapter intentionally does not fabricate calls, append
answers, or repair truncated output.
