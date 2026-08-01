# EvalScope deployment-backed environments

## Current readiness

The native local/OpenAI-compatible Agent path is already validated on
General-FC and BFCL-v3.  The BFCL-v3 full run completed 4,441/4,441 samples
with the official EvalScope report and no provider errors.  The K2 correction
run uses the registered `k2_verifier` name and is tracked separately.

This repository now carries two deployment profiles:

| Profile | Official task runtime | Scope | Status |
| --- | --- | --- | --- |
| `swebench_verified_mini` | one Docker sandbox per SWE-bench instance | 50 tasks | prepared; preflight first |
| `terminal_bench_v2_1` | Harbor/terminus-2 Docker environment | 89 tasks | prepared; preflight first |

The full SWE-bench Verified, Multilingual, and Pro suites are intentionally
not included in the first deployment pass.  They require substantially more
Docker storage and longer task lifetimes.  The remote Docker host currently
has about 99 GB free and already stores a large image cache, so image creation
must be observed rather than started blindly.

## Environment preflight

Run the launcher with `PREPARE_ONLY=1`.  This checks `uv`, Docker CLI, Docker
daemon, the selected profile, and Docker storage without starting a model
request or removing any Docker state:

```bash
PREPARE_ONLY=1 \
scripts/run_evalscope_docker_benchmark.sh \
  swebench_verified_mini MODEL_ALIAS http://127.0.0.1:29533/v1 docker-preflight
```

For Terminal-Bench, use `terminal_bench_v2_1` as the first argument.  The
launcher uses `uv --no-default-groups --group ... --no-sync`; it does not mix
the normal Agent dependency group with the SWE-bench or Terminal-Bench group.

## First real run

Start with one official task after the preflight and preserve its complete
`predictions/`, `reviews/`, `reports/`, and `command.log` artifacts:

```bash
scripts/run_evalscope_docker_benchmark.sh \
  swebench_verified_mini MODEL_ALIAS http://127.0.0.1:29533/v1 swe-mini-smoke 1
```

The `1` is a deployment smoke test, not a benchmark score.  Remove the limit
only after the Docker image pull/build, tool trajectory, cleanup, and official
review have all been checked.  Terminal-Bench is launched with
`--no-agent-config` because its official Harbor/terminus-2 loop owns the tool
protocol; SWE-bench uses the official `swe_bench_toolcall` native strategy.

No output is repaired in this layer, no scorer is replaced, and no model
service is started or stopped by these files.
