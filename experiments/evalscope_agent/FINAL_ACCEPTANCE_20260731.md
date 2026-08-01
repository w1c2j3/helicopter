# EvalScope Agent 流水线最终验收审计（候选版本）

状态：**候选版本，未达到正式发布门槛**。本文件以及本轮重测记录只保存在本地，不提交、不上传 GitHub。

## 固定运行条件

- 分支：`updata/supported-dataset`
- 7.2B：`http://127.0.0.1:29572/v1`，模型 `rwkv7-g1h-7.2b-20260710-ctx10240`
- 13.3B：`http://127.0.0.1:29534/v1`，模型 `rwkv7-g1h-13.3b-20260710-ctx10240`
- API key：`rwkv-skills`
- 上下文：10240 tokens
- 温度：0
- 生成上限：2048 tokens
- Agent 专项依赖：`uv --no-default-groups --group agent`

## 验收项

| 验收项 | 状态 | 证据/说明 |
| --- | --- | --- |
| 本地 RWKV 接口可调用 | PASS | 两个端点均完成真实 HTTP 请求；native tool-call 由服务返回 |
| naive Chat 适配 | PASS | 保留原消息语义、顺序和系统提示词；适配只发生在发送前 |
| 原始响应保留 | PASS | predictions、trace、acceptance report 均保留原始模型响应 |
| 严格答案提取 | PASS | 无法提取时显式失败；不补写选项、字段、结束符或答案 |
| 严格判别 | PASS | 区分 `format_invalid`、`extraction_failed`、`model_error`、`correct_no_tool_call` |
| 运行级错误追踪 | PASS | acceptance report 顶层 `run_errors` 保留第三方 adapter、认证、上下文和超时日志片段 |
| Agent 回归测试 | PASS | `63 passed, 2 warnings` |
| 完整输入到报告流水线 | PASS | `helicopter eval evalscope` 生成 predictions/reviews/reports/traces/acceptance report |
| 关键能力指标达到预设阈值 | NOT MET | Agent 闭环、停止、失败恢复和最终回答能力仍有明显失败 |
| 全部目录均已取得有效分数 | NOT MET | BFCL-v4、Tau2/Tau3、MCP 服务类和环境类 benchmark 仍不可计分 |
| 正式发布 | NO-GO | 依据上述未满足项，不标记正式可用版本 |

## 真实分数

| Benchmark | 7.2B | 13.3B |
| --- | ---: | ---: |
| BFCL-v3 非多轮（3641） | 39.11% | 45.84% |
| GAIA（165） | 3.64%（6/165） | 1.21%（2/165） |
| OfficeQA（133） | 1.50%（2/133） | 2.26%（3/133） |
| General-FC F1（2000） | 58.52% | 47.80% |
| K2-Verifier F1（2000） | 58.32% | 48.02% |
| ACEBench（1023） | 31.59% | 37.69% |
| ToolBench Action EM | 9.65% | 16.02% |
| ToolBench Primary F1 | 4.68% | 9.55% |

验证器结果单独记录，不并入模型能力总分：

- Kimi-Verifier：两模型 inference error 均为 0/55；默认参数接受率 100%。
- MiniMax-Verifier：tool-call 匹配率 35.35% / 33.33%，schema 准确率 75.00% / 72.22%（7.2B / 13.3B）。

## 未计分项目和原因

- BFCL-v4：memory prerequisite 的输入至少 8193 tokens，再请求 2048 输出，超过 10240，服务返回 HTTP 400。
- BFCL-v3 多轮：正式 BFCL-v3 分数只覆盖非多轮 3641 题，未把多轮空项当作 0 分。
- Tau2-Bench：278 题数据加载成功且本地模型真实调用，但 Tau2 多轮路径得到 `completion.usage=None` 时，EvalScope 1.9.1 adapter 无条件调用 `model_dump()`；对照请求证明 RWKV 服务和普通 EvalScope 请求可以返回 usage，故归类为第三方 Tau2 路径的契约错误。
- Tau3：需要下载 866 个知识库文件，smoke 在数据下载阶段无进展后停止。
- BrowseComp：需要真实浏览/搜索 agent 和 judge，不具备当前本地工具条件。
- GDPval：依赖 Docker 交付物环境，官方质量评分在外部 judge 完成。
- MCP-Atlas：需要外部 MCP-Atlas 服务。
- ResearchRubrics：需要额外 judge 模型，不属于当前纯请求/提取/判别链路。

## 回退节点

已有主要提交和标签记录在 `experiments/evalscope_agent/RESULTS.md`，包括：

- `baseline/pre-evalscope`
- `baseline/wsl-actual`
- `evalscope/native-local-code`
- `evalscope/native-local-benchmark`
- `evalscope/native-local-gaia`
- `evalscope/forwarded-external-code`
- `evalscope/reproducible-runner`
- 当前分支最新提交：`30e0c6e fix(evalscope): apply request timeout to OpenAI client`

## 回归命令

`agent` 与 `eval` uv 依赖组互斥，Agent 专项使用：

```bash
uv lock --check
uv run --no-default-groups --group agent --with pytest \
  pytest -q tests/test_naive_chat.py \
    tests/test_evalscope_agent.py \
    tests/test_evalscope_agent_results.py \
    tests/test_summarize_evalscope_agent_matrix.py \
    tests/test_parallel_candidate_proxy.py
```

结果：**63 passed，2 warnings**。

结论：代码层集成、消息适配、原始响应追踪、严格提取/判别和可复现运行已经具备；模型层仍未稳定完成“工具调用 → 获取结果 → 判断充分性 → 停止并回答”的 Agent 闭环，因此当前只能作为候选测评流水线，不能宣称正式通过。

## 2026-07-31 continuation audit

- WideSearch smoke: 7.2B and 13.3B both `0/2`; not a formal score. EvalScope exposed `bash` plus automatic submit, while the model emitted text `web_search` JSON and no native `tool_calls`.
- SWE-Bench Verified Mini smoke: still no score. Docker Hub pull failed with HTTP 500/`unexpected EOF`; fallback image creation started, but the Django instance remained in `conda create -n testbed python=3.6` before the first model request.
- Current formal score inventory: five end-to-end Agent-capability results for both models (BFCL-v3 non-multi, GAIA, OfficeQA, General-FC, K2-Verifier), plus complete verifier/static metric reports kept separate from the capability table.

- Tau2 compatibility smoke (not a formal score): an experiment-only `usage=None` shim and local RWKV user simulator allowed the run to reach native tool calls and environment results. The first multi-turn task then exceeded the fixed context budget (`8193` input + `2048` output > `10240`) and the service returned HTTP 400. The shim and runner are intentionally untracked and were not pushed.

- Version audit: `updata/supported-dataset` and `origin/updata/supported-dataset` both point to `4ad3787`. Major EvalScope checkpoints remain independently recoverable through the existing `evalscope/*` tags, including `evalscope/chat-adapter`, `evalscope/agent-matrix-runner`, `evalscope/run-error-diagnostics`, `evalscope/judge-config`, and `evalscope/wide-search-deps`.

## Agent registry disposition

The EvalScope 1.9.1 registry contains 27 entries with category `agent`. The following are the current dispositions after checking their adapters and runtime requirements:

- Completed with formal reports: `bfcl_v3` (non-multi scope), `gaia`, `general_fc`, `k2_verifier`, `kimi_verifier`, `minimax_verifier`, `officeqa`.
- Completed diagnostic only: `wide_search` (2 samples per model; no native tool call because the task contract exposed `bash`).
- Context/adapter compatibility failure: `bfcl_v4` memory snapshots exceed the 10240-token service window with the fixed 2048-token output; `tau2_bench` reaches `completion.usage is None` in the third-party EvalScope adapter; `tau3_bench` stopped during its large knowledge-base download; `tau_bench` uses the same Tau adapter family.
- External service or judge required: `browsecomp`, `gdpval`, `mcp_atlas`, `researchrubrics`, `toolathlon`.
- Docker/task repository or multimodal environment required: `claw_eval`, `deep_swe`, `skillsbench`, `swe_bench_lite_agentic`, `swe_bench_multilingual_agentic`, `swe_bench_pro`, `swe_bench_verified_agentic`, `swe_bench_verified_mini_agentic`, `terminal_bench_v2`, `terminal_bench_v2_1`.

## Current continuation addendum (authoritative)

Earlier continuation paragraphs in this local draft contain historical metadata; the values below supersede them.

- Current branch and origin: `updata/supported-dataset` at `04f1232` (`fix(evalscope): parse and classify BFCL-v4 outputs`).
- Current Agent regression: `69 passed, 2 warnings` with `uv lock --check`.
- Formal BFCL-v4 main short configuration: 7.2B `48.54% (800/1648)`; 13.3B `61.04% (1006/1648)`; both acceptance reports have `format_invalid=0` and no run errors.
- BFCL-v4 additional diagnostic runs are intentionally not scores. At the latest audit, 7.2B was `1442/5256`; 13.3B was `297/5256` and had third-party `KeyError: 0` on the first 297 samples. This is adapter/data-shape incompatibility, not a context overflow.
- No experiment artifacts in this file or the companion retest record are staged or uploaded.
- Live component follow-up: 7.2B `23/56` and 13.3B `24/56`; both official BFCL component reports give `LIVE=20.00%` and weighted `Overall=2.00%` for this Live-only run.
- Agentic Web Search follow-up: both models completed `web_search_base` (100 samples) with official `0.00%`; both acceptance reports classify all 100 outputs as `format_invalid`. The lower-level trace shows ordinary text plus `tool_calls=[]`, while the synthesized raw response is `[[[]]]`; this is a model/protocol failure, not answer-extractor repair.
- Native cross-check: the identical first Web Search request sent directly to 29572 returned `finish_reason=tool_calls` with a valid `search_engine_query`. Native full reruns stalled in external-tool execution and were stopped at 2/100 (7.2B) and 3/100 (13.3B); no partial score is accepted.
- Memory follow-up: `memory_kv`, `memory_vector`, and `memory_rec_sum` each contain 192 entries with 37 prerequisite IDs; serialized entries reach 21,891 characters. They were skipped under the fixed 10,240-token context/2,048-token output budget and recorded as context-risk cases, not scored.
- Multi-Turn follow-up: the three non-long-context subsets loaded 600 samples per model, but the router fan-out made the first items take 93--214 seconds. Runs were stopped at 3/600 (7.2B) and 1/600 (13.3B); no partial score is accepted. Logs and raw traces remain remote and local experiment records remain untracked.

## Native tool-call fallback continuation

- Code checkpoint: commit `6a1289b` and tag `evalscope/bfcl-v4-native-tool-fallback`, pushed to `origin/updata/supported-dataset`.
- Confirmed issue: when every parallel candidate was unparseable, the router synthesized ordinary text with `tool_calls=[]`, discarding a native upstream tool call that the same request could produce directly.
- Fix: on candidate/aggregate parse failure, forward the original Chat request unchanged to `/v1/chat/completions` and preserve its response and `tool_calls`; if that fallback fails, retain the strict invalid-result path and record the error.
- Regression: the local upstream fixture verifies two failed candidate calls followed by an unchanged native Chat payload and a preserved `finish_reason=tool_calls`; the related suite now passes `70 passed, 2 warnings` under `uv`.
- This is an adapter correctness fix, not a new benchmark score. Existing Web Search and Multi-Turn partial runs remain excluded from scoring; no experiment artifacts were staged or uploaded.

## Official-score policy and latest BFCL-v4 rerun

The EvalScope official sandbox/scorer remains the only source of formal benchmark scores. The local extraction and acceptance reports preserve raw responses and classify failures, but do not rewrite or replace the official score.

The latest fixed short BFCL-v4 run produced official top-level scores of `51.21%` for G1h 7.2B and `62.26%` for G1h 13.3B, each over 1648 samples. The report aggregate fields are `OVERALL=11.29%` and `OVERALL=14.00%`, respectively; these are partial aggregates because Agentic and Multi-Turn were not included and are not full formula-based BFCL-v4 overall scores. The complete official report files remain on the remote host under the `bfcl-v4-*-native-fallback/reports/.../bfcl_v4.json` paths.

The experiment records are intentionally untracked and were not uploaded to GitHub.
