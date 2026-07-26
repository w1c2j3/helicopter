# EvalScope Agent integration experiment record

All model-backed commands in this record were run inside WSL with the
repository environment and `uv run --no-default-groups --no-sync`. The fixed
endpoint was `http://127.0.0.1:19329/v1`, model
`rwkv7-g1h-2.9b-20260710-ctx10240`, API key `rwkv-skills`, and context limit
10240 tokens.

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

The second run proves the complete path: ModelScope dataset loading, proxy
serialization, local model call, unchanged response, EvalScope report, and
Helicopter `raw/trace_report.json`. The score remains zero because this model
response contains prose and no OpenAI `tool_calls`; it is not an extractor or
discriminator success case.

The GAIA run additionally proved EvalScope's native AgentLoop path and was
rerun after installing `evalscope[sandbox]` with `uv pip`. Its fixed subset is
recorded in `gaia-level1.json`. The model repeatedly described the intended
bash action instead of emitting a tool call and reached the 1024-token cap;
the official score is therefore zero and the raw two-request trace is kept.

## Failure classification

- Chat format: fixed for the local endpoint by the naive Chat proxy; verified
  with a real HTTP 200 response.
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
experiment window. Native function-calling datasets require a serving process
with an actual tool-call parser, which the current 19329 tunnel does not have.
GAIA can now enter the AgentLoop after the sandbox extra is installed, but the
model still fails its required tool-call format. These are recorded blockers,
not silently converted to passing results.
