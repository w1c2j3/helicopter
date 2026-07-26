# EvalScope Agent integration baseline

This directory records the unmodified-code baseline for the EvalScope Agent
integration. The baseline commit is `4a68cb0366875930183bd9820436a7f81e223f7e`.

## Fixed target configuration

- API base URL: `http://127.0.0.1:19329/v1`
- Model: `rwkv7-g1h-2.9b-20260710-ctx10240`
- API key: `rwkv-skills` (never stored in artifacts)
- Maximum context: `10240` tokens
- Windows preflight runtime: Python `3.12.11`, `uv 0.7.19`.
- Actual model runtime: WSL Python `3.12.3`, `uv 0.9.24`.
- Source branch: `chase/supported-dataset`

The representative Agent benchmark set selected for the later fixed-condition
comparison is `bfcl_v3`, `tau2_bench`, `gaia`, `swe_bench_lite_agentic`, and
`terminal_bench_v2`. The current unmodified CLI has no EvalScope Agent runner,
so those runs are recorded as not started rather than being presented as
baseline scores.

## Baseline status

The first Windows-side model preflight was refused by the local host; the
complete redacted request and error are in `raw-api-preflight.json`. The
endpoint is a WSL-local SSH tunnel, so the reproducible baseline was then
rechecked inside WSL with the repository's `uv` runtime. That recheck is in
`chat-preflight.json` and records the unmodified tool-call request returning
HTTP 400 because the server was not started with an auto tool-call parser.
`chat-naive-preflight.json` records the same semantic system/user content
serialized as naive Chat and returning HTTP 200. The old pipeline's
`bfcl_simple_python` two-sample probe still recorded score `0.0000` with
`HTTP Error 400: Bad Request`; the response body was not preserved by that
old path. These are baseline failures, not corrected scores.

The existing code was not changed to work around this blocker. The current
test evidence is recorded in `test-results.txt`:

- LightEval answer adapter subset: `21 passed`.
- Existing sampling/config subset: `1 passed, 1 failed`; the failure resolves
  `max_tokens` to `2048` while the test expects `8192`.
- Full test collection: blocked by missing NLTK `punkt` and `punkt_tab`
  resources in the environment. An isolated `uv` download attempt timed out.
- Syncing the existing `eval` dependency group is separately blocked by
  `natto-py==1.0.1` importing removed Python 3.12 `distutils`; this is not
  changed in the baseline.

## Reproduction

Use a temporary environment so the repository's existing `.venv` is not
modified:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:TEMP\helicopter-uv-baseline"
$env:SETUPTOOLS_USE_DISTUTILS = "local"
uv sync --no-default-groups --python 3.12
uv pip install --python "$env:UV_PROJECT_ENVIRONMENT\Scripts\python.exe" --editable .\src\eval\lighteval pytest requests
uv run --no-default-groups --no-sync -- pytest -q tests/test_lighteval_answer_adapters.py
```

For the actual WSL endpoint, run from the WSL checkout so `127.0.0.1:19329`
resolves to the model tunnel:

```sh
cd /home/chase/GitHub/helicopter
export HELICOPTER_EVAL_API_KEY=rwkv-skills
uv run --no-default-groups --no-sync python \
  experiments/evalscope_agent/baseline/probe_chat.py \
  --output experiments/evalscope_agent/baseline/chat-preflight.json
```
