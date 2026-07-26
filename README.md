# Helicopter

Helicopter is an RWKV leaderboard-run framework. It coordinates external RWKV
vLLM endpoints, verl-based training, and benchmark-oriented experiment runs
with a small CLI. The vllm-rwkv inference engine is maintained as an independent
checkout and is not vendored in this repository.

The current focus is RWKV7:

- `infer`: start a vLLM server for an RWKV checkpoint.
- `eval`: run LightEval tasks against an OpenAI-compatible endpoint, or run a
  managed one-shot evaluation that starts and stops vLLM automatically.
- `takeoff`: start verl training for an RWKV checkpoint. The supported takeoff
  path is GRPO.
- `scripts/install_remote.sh`: prepare the BBT DevPod GPU workspace, sync this
  repository, and run the local installer remotely.
- `scripts/install_local.sh`: create/update the project `.venv`, install the
  declared RWKV dependency group, and install local editable `rwkv-lm` and
  `verl` packages. Install vllm-rwkv in its own checkout/environment.

## Repository layout

```text
configs/
  example.toml              # public example experiment config
  local/*.toml              # machine-local experiment configs
scripts/
  install_local.sh          # prepare the current machine/workspace
  install_remote.sh         # sync and prepare the remote DevPod workspace
src/cli/helicopter_cli/     # Python CLI package
src/scoreboard-server/      # FastAPI scoreboard API and PostgreSQL store
src/scoreboard-client/      # Next.js scoreboard UI
src/train/rwkv-lm/          # RWKV training code
src/train/verl-rwkv/        # verl RWKV integration
```

Set `[paths].vllm_rwkv_path` or `HELICOPTER_VLLM_RWKV_PATH` only for workflows
such as `takeoff` that import an external vllm-rwkv checkout. Endpoint-based
evaluation does not import or install vllm-rwkv.

`AGENTS.md` is intentionally ignored in this repository because it may contain
machine-specific remote connection details. Use `.env.example` and
`AGENTS.example.md` as public templates.

## Environment files and configs

Copy `.env.example` to a private env file before running commands:

```bash
cp .env.example .env.local
```

For remote DevPod use, keep the private remote values in `.env.remote`.

The env files use simple dotenv syntax:

```text
KEY=value
export KEY=value
```

Do not put shell expressions in env files that the Python CLI must read. Values
already present in the command environment override values from `.env.local` or
`.env.remote`, which makes command-scoped overrides predictable:

```bash
WEIGHT_PATH=/workspace/Weights/RWKV helicopter infer g1g-1.5b
```

Experiment settings live in TOML files. If `--config` is omitted, the CLI uses
the newest `configs/local/*.toml`; otherwise it falls back to
`configs/example.toml`.

Important config sections:

- `[models.<name>]`: maps a CLI model alias to a checkpoint file or path.
- `[datasets.<name>]`: maps a dataset alias to a dataset root.
- `[infer]`: vLLM serving defaults.
- `[lighteval]`: LightEval endpoint, output, and custom-task defaults.
- `[function_calling]`: native OpenAI `tool_calls` benchmark defaults.
- `[agent_harness]`: external agent harness planning defaults.
- `[evalscope]`: EvalScope Agent dataset, AgentLoop/Bridge, sandbox, and output defaults.
- `[takeoff.grpo]`: verl GRPO training defaults.

Scoreboard database settings are read from `SCOREBOARD_DB_*` first and then
from standard `PG*` variables. This matters for `helicopter eval run
--scoreboard`, because the CLI loads dotenv files before writing results into
the scoreboard database:

```text
SCOREBOARD_DB_HOST=/var/run/postgresql
SCOREBOARD_DB_PORT=5432
SCOREBOARD_DB_USER=postgres
SCOREBOARD_DB_NAME=helicopter
```

## Prepare the environment

Remote preparation is the expected path for full RWKV vLLM/verl work:

```bash
scripts/install_remote.sh
```

The remote installer:

- validates the target DevPod Pod and node;
- checks that the running container uses the required runtime image;
- syncs this repository with `rsync`;
- preserves the remote `.venv`;
- runs `scripts/install_local.sh` inside the remote repo path.

For local or already-synced workspace preparation:

```bash
scripts/install_local.sh
```

Useful install overrides:

```bash
VLLM_REBUILD=1 scripts/install_local.sh
VERL_REINSTALL=1 scripts/install_local.sh
INSTALL_PROFILE=full scripts/install_local.sh
```

Use `DRY_RUN=1` to print installer actions without executing them:

```bash
DRY_RUN=1 scripts/install_remote.sh
```

## CLI usage

Run the CLI through the installed console script:

```bash
helicopter --help
```

During development, the package can also be run directly:

```bash
PYTHONPATH=src/cli python3 -m helicopter_cli --help
```

### Start RWKV vLLM serving

Dry-run first to inspect the exact command and environment:

```bash
helicopter infer --config configs/example.toml --dry-run g1g-1.5b
```

Start the server:

```bash
helicopter infer --config configs/example.toml g1g-1.5b
```

Override serving parameters from the CLI when an experiment explicitly needs
them:

```bash
helicopter infer g1g-7.2b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 2048 \
  --max-num-batched-tokens 65536
```

RWKV vLLM uses upstream defaults by default. For RWKV7, set only WKV mode unless
you are debugging a specific vLLM issue. GRPO `takeoff` keeps embedding
preprocessing on GPU with `HELICOPTER_TAKEOFF_EMB_DEVICE=gpu`:

```bash
VLLM_RWKV7_WKV_MODE=fp32io16 helicopter infer g1g-1.5b
```

### Run LightEval

Use `eval lighteval` when a compatible endpoint is already running:

```bash
helicopter eval lighteval \
  --config configs/example.toml \
  g1g-1.5b "gsm8k|0" \
  --max-samples 2
```

Use `eval run` for a managed one-shot evaluation. For a local endpoint URL, the
command starts vLLM in the background, waits for `/v1/models`, runs LightEval,
writes a performance report, optionally records scores into the scoreboard
database, and stops the managed server:

```bash
helicopter eval run \
  --config configs/example.toml \
  g1d-0.4b "gsm8k|0" \
  --max-samples 2 \
  --scoreboard
```

Useful run-control options:

```bash
helicopter eval run g1d-0.4b "gsm8k|0" --dry-run
helicopter eval run g1d-0.4b "gsm8k|0" --no-server
helicopter eval run g1d-0.4b "gsm8k|0" --keep-server
helicopter eval run g1d-0.4b "gsm8k|0" --server-timeout 900
```

`eval run` forwards the same LightEval options as `eval lighteval`, plus serving
overrides such as `--wkv-mode`, `--emb-device`,
`--gpu-memory-utilization`, `--max-num-seqs`, and
`--max-num-batched-tokens`. The task argument can be omitted when
`[lighteval].tasks` is set in the selected config.

### Run Raw Performance Profiles

Use `eval perf` when you want a service-level throughput probe without running a
formal benchmark. It sends raw OpenAI `/v1/completions` requests so prefill and
decode behavior can be measured separately from LightEval or function-calling
score logic:

```bash
helicopter eval perf \
  --config configs/example.toml \
  g1d-0.4b \
  --base-url http://127.0.0.1:8000/v1 \
  --profile decode \
  --prompt-tokens 128 \
  --output-tokens 256 \
  --requests 64 \
  --concurrency 8 \
  --ignore-eos \
  --output results/performance/g1d_decode.json
```

`--profile prefill` defaults to a longer prompt and short generation;
`--profile decode` defaults to a shorter prompt and longer generation. The JSON
report includes request throughput, prompt/completion/total token throughput,
E2E latency percentiles, and per-error counts. Start the vLLM server separately
with `helicopter infer` or point `--base-url` at an existing service.

### Run Batched Evaluations

Use `eval batch` when one command should run a selected set of model/benchmark
pairs. Each `--job` is one logical model selection; separate jobs can select
different benchmarks, so the command does not silently create a full
model-by-benchmark matrix:

```bash
helicopter eval batch \
  --config configs/example.toml \
  --job 'g1d-0.4b=gsm8k|0,mmlu|0' \
  --job 'g1g-1.5b=gsm8k|0' \
  --job 'g1d-0.4b=fc:bfcl_v3' \
  --gpus 0,1 \
  --parallel 2 \
  --scoreboard \
  --wkv-mode fp16 \
  --emb-device gpu \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768
```

Use `fc:TASK` inside `--job` for a native function-calling benchmark. The
older `--models`/`--tasks`/`--fc-tasks` options remain supported for config
driven matrix runs.

When `--scoreboard` is set, batch mode skips benchmarks that already have a
score unless `--rerun` is passed. Use `--batch-output path/to/report.json` to
choose a stable report path; otherwise real runs write under
`results/eval_batch/`. Dry-runs print the child `eval run` or function-calling
plans without writing a report unless `--batch-output` is explicitly provided.

### Run Function Calling

Function-calling benchmarks use the native OpenAI-compatible `tools` request and
`message.tool_calls` response path. They are not registered as LightEval custom
tasks, so there is only one FC score path:

```bash
helicopter eval function-calling \
  --config configs/example.toml \
  g1d-0.4b bfcl_v3 \
  --max-samples 2 \
  --scoreboard
```

Use `all` or omit the task argument to run every native FC task. Supported task
ids are BFCL, APIBank, ComplexFuncBench, and ToolAlpaca variants. Runtime knobs
such as token cap, request timeout, concurrency, and managed-server timeout live
under `[function_calling]` or `HELICOPTER_FC_*` environment variables; the CLI
surface intentionally stays small. Managed local runs automatically start vLLM
with `--enable-auto-tool-choice`; when launched by `eval batch`, the batch
serving overrides are forwarded to the managed FC server as well.

Inspect the available custom tasks and metric status before treating results as
formal scores:

```bash
helicopter eval lighteval-tasks export --contains gsm8k --format text
helicopter eval lighteval-tasks judges --format summary
```

Some custom tasks intentionally use proxy or sanity metrics rather than the
official benchmark harness. Examples include Arena-Hard baseline token F1,
SWE-Bench patch token F1 or nonempty checks, and TAU static-plan token F1.
`lighteval-tasks judges` marks these cases explicitly.

### Run EvalScope Agent benchmarks

EvalScope Agent benchmarks use the same OpenAI-compatible vLLM endpoint but keep
their own multi-turn tool trace and report format. Install only the optional
Agent dependency group first; the full `eval` and `rwkv` groups are not needed
for this path:

```bash
uv sync --no-default-groups --group agent
```

List the Agent datasets tracked from EvalScope's supported-dataset registry:

```bash
helicopter eval evalscope --list-datasets
```

Run a Native AgentLoop benchmark against the requested WSL-local RWKV
endpoint. The command starts a local naive-Chat proxy by default; it records
the original request, converted request, raw response, and a diagnostic trace
report under `results/evalscope`:

```bash
helicopter eval evalscope \
  --config configs/example.toml \
  g1h-2.9b general_fc \
  --model-catalog configs/models/g1h-single-replica.toml \
  --base-url http://127.0.0.1:19329/v1 \
  --api-key rwkv-skills \
  --no-server \
  --limit 1 \
  --strategy function_calling \
  --agent-environment local
```

For coding agents whose server does not expose OpenAI tool calls, use
EvalScope's text strategy and install its optional SWE-bench extra separately:

```bash
uv pip install --python .venv/bin/python 'evalscope[swe_bench]==1.9.1'
helicopter eval evalscope g1h-2.9b swe_bench_verified_agentic \
  --model-catalog configs/models/g1h-single-replica.toml \
  --base-url http://127.0.0.1:19329/v1 --api-key rwkv-skills \
  --no-server --strategy swe_bench_backticks --agent-environment local --limit 1
```

`function_calling` requires the serving process to support an OpenAI
`tool_calls` response. The naive-Chat adapter removes unsupported tool fields
only at the outbound transport boundary and never fabricates a tool call or
answer. A missing tool call is recorded as `format_invalid`; HTTP failures and
context-limit stops remain separately visible in `raw/trace_report.json`.

For a regular Agent dataset, choose the tools explicitly:

```bash
helicopter eval evalscope g1d-0.4b gaia \
  --mode native \
  --tools python_exec \
  --agent-environment docker \
  --max-steps 10 \
  --limit 5
```

EvalScope can also drive an external CLI such as Codex or Claude Code through
its Agent Bridge:

```bash
helicopter eval evalscope g1d-0.4b swe_bench_pro \
  --mode bridge \
  --framework codex \
  --agent-environment docker \
  --agent-timeout 1800
```

The command automatically starts and stops the configured local vLLM server
unless `--no-server` or `--keep-server` is set. Reports, predictions, reviews,
and Agent traces are written under `results/evalscope` by default. EvalScope's
official sandbox/verifier owns the benchmark score; the local diagnostic layer
keeps raw responses and strict extraction/discrimination results separate from
the single-turn LightEval and native function-calling score paths.

The curated directly runnable non-function-calling LightEval catalog is stored
in the scoreboard database table `benchmark_catalog`. It keeps 30 diverse,
recognized public benchmark rows each for math, coding/CS, instruction/task
following, and knowledge. The allowlist is generated from common task families
available in the pinned LightEval source, such as MATH, GSM8K/MGSM, AIME,
OlympiadBench, LiveCodeBench and CS exam suites, IFEval/IFBench/BBH/BIG-Bench,
MMLU/GPQA/ARC, TruthfulQA/OpenBookQA, and Natural Questions/TriviaQA/SQuAD-style
QA; agent, tool-use, function
calling, and endpoint-incompatible perplexity suites stay out of this direct
HF/LightEval catalog.

```bash
uv run --group eval python scripts/seed_non_fc_lighteval_benchmarks_db.py
uv run --group eval python scripts/verify_non_fc_lighteval_benchmarks_db.py
helicopter eval batch --tasks-from-db --scoreboard --models g1g-1.5b
```

The agent benchmark scope is tracked separately from the runnable LightEval task
registry:

```bash
helicopter eval lighteval-tasks coverage \
  --source benchmarks/agent_benchmarks.json \
  --format summary
helicopter eval lighteval-tasks coverage \
  --source benchmarks/agent_benchmarks.json \
  --format jsonl
```

Only rows with direct LightEval coverage can be run through `eval run`. Rows
marked in the source metadata as `external_harness_required` need their official
agent harness before they should be treated as reproducible agent scores.
The source file groups agent benchmarks into five run-planning pipelines:
`coding_agent`, `search_agent`, `tool_mcp_agent`,
`office_enterprise_workflow_agent`, and `stem_tool_agent`. Reasoning, math,
context-learning, and long-context benchmarks are kept under `excluded` unless
they require tool-using agent behavior.

Prepare external agent harnesses separately from LightEval:

```bash
helicopter eval agent-harness list --format text
helicopter eval agent-harness preflight --pipeline coding_agent --strict
helicopter eval agent-harness plan swe_bench_verified \
  --model g1d-0.4b \
  --output-dir results/agent_harness
helicopter eval agent-harness convert swe_bench_verified \
  --input results/agent_harness/swe_bench_verified/rwkv_outputs.jsonl \
  --output results/agent_harness/swe_bench_verified/predictions.jsonl \
  --model g1d-0.4b
```

`agent-harness run` is intentionally conservative. It only executes benchmark
profiles with an implemented Helicopter runner. External official harnesses write
a plan artifact and exit nonzero instead of producing fake scores:

```bash
helicopter eval agent-harness run deepswe \
  --model g1d-0.4b \
  --output-dir results/agent_harness
```

BrowseComp currently has an answer-only local proxy through LightEval. Because
that is not a browser-runtime agent score, it requires an explicit opt-in:

```bash
helicopter eval agent-harness run browsecomp \
  --model g1d-0.4b \
  --base-url http://127.0.0.1:8000/v1 \
  --no-server \
  --allow-proxy \
  --max-samples 2
```

The agent harness commands do not rewrite official sandboxes. They record which
official harness should own execution and verification. For example, SWE-bench
planning emits the patch-prediction artifact expected by the official Docker
harness, while Terminal-Bench planning marks the OpenAI-compatible terminal
agent adapter as the next required layer before `tb run` or `harbor run` should
be used.
The `convert` step is the middle-format boundary: RWKV or Helicopter raw output
is normalized to `helicopter_agent_v1` internally, then exported to the official
artifact shape. For SWE-bench that artifact is `predictions.jsonl` with
`instance_id`, `model_name_or_path`, and `model_patch`.

### Start GRPO takeoff training

Dry-run a GSM8K GRPO run:

```bash
helicopter takeoff \
  --config configs/example.toml \
  --dry-run \
  --dataset gsm8k \
  g1g-1.5b grpo
```

Start the run:

```bash
helicopter takeoff \
  --config configs/example.toml \
  --dataset gsm8k \
  g1g-1.5b grpo
```

Pass extra Hydra overrides to the underlying verl entrypoint:

```bash
helicopter takeoff g1g-1.5b grpo \
  --dataset gsm8k \
  --override trainer.total_epochs=1 \
  --override trainer.save_freq=10
```

`takeoff` requires the project Python executable to exist. By default it uses
the configured `.venv/bin/python`; set `HELICOPTER_PYTHON` or `paths.python` only
when an explicit override is intended:

```bash
HELICOPTER_PYTHON=/workspace/Projects/MachineLearning/helicopter/.venv/bin/python \
helicopter takeoff --dataset gsm8k g1g-1.5b grpo
```

## Common command-scoped overrides

```bash
WEIGHT_PATH=/workspace/Weights/RWKV
DATASETS_PATH=/workspace/Datasets
HELICOPTER_NUM_NODES=1
HELICOPTER_NUM_DEVICES=8
HELICOPTER_TAKEOFF_WKV_MODE=fp32io16
HELICOPTER_TAKEOFF_EMB_DEVICE=gpu
```

Keep checkpoint files, datasets, `.env.local`, `.env.remote`, and machine-local
agent notes out of the public repository.

## Lightweight checks

The root CLI has standard-library tests and does not require the full RWKV
dependency group:

```bash
PYTHONPATH=src/cli python3 -m unittest tests.test_cli
PYTHONPATH=src/cli python3 -m compileall -q src/cli/helicopter_cli tests
bash -n scripts/install_local.sh scripts/install_remote.sh
```
