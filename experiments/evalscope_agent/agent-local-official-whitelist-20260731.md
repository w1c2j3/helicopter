# EvalScope Agent local-official whitelist — 2026-07-31

This run measures only datasets that are registered in the installed EvalScope
Agent package and whose scoring can run without a remote service, an LLM judge,
Docker/task sandbox, a code repository, or a special external environment.

## Full-run whitelist

- `bfcl_v3` — official BFCL rule scorer
- `bfcl_v4` — official BFCL-v4 scorer; the current full official DB result is reused
- `general_fc` — official function-calling scorer
- `k2_verifier` — official vendor-verifier scorer
- `kimi_verifier` — synthetic official parameter-compliance scorer
- `minimax_verifier` — official vendor-verifier scorer

## Explicit exclusions

- `tau_bench`, `tau2_bench`, `tau3_bench`: third-party Tau environment/package and
  user-simulation path; not a self-contained local run in this environment.
- `officeqa`: requires the large Treasury Bulletin corpus and a sandboxed bash
  document environment.
- `gaia`, `wide_search`, `browsecomp`, `deepsearchqa`, `researchrubrics`, and
  `toolathlon`: retrieval/web/service or special dataset infrastructure.
- SWE-bench, Terminal-Bench, DeepSWE, SkillsBench, and Claw-Eval: Docker,
  repository, code-task, or multimodal special environments.
- `automation_bench` and `job_bench`: present in the project catalog but not
  registered by the installed EvalScope package; they cannot produce an official
  EvalScope score in this pinned environment.

The importer stores EvalScope's raw report, prediction, review, and sample score
value. It does not rescore or repair model output. If an adapter exposes only
aggregate validator values, the official aggregate remains authoritative and the
database records that the sample value was not reducible to a pass/fail row.
