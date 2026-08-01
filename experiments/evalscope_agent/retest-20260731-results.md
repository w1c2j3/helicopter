# EvalScope Agent 重测记录（2026-07-31）

本文件是本轮远端模型重测的本地审计记录，**不加入 Git、不上传 GitHub**。
模型接口为：

- G1h 7.2B：`http://127.0.0.1:29572/v1`
- G1h 13.3B：`http://127.0.0.1:29534/v1`
- API key：`rwkv-skills`
- 上下文上限：10240 tokens
- 生成上限：2048 tokens，temperature=0

## 可计分结果

| Benchmark | G1h 7.2B | G1h 13.3B | 备注 |
| --- | ---: | ---: | --- |
| BFCL-v3 非多轮 | 39.11%（3641） | 45.84%（3641） | 不包含 800 个多轮样本 |
| GAIA | 3.64%（6/165） | 1.21%（2/165） | 完整 165 题 |
| OfficeQA | 1.50%（2/133） | 2.26%（3/133） | 完整 133 题 |
| General-FC F1 | 58.52%（2000） | 47.80%（2000） | F1，不是准确率 |
| K2-Verifier F1 | 58.32%（2000） | 48.02%（2000） | F1，不是准确率 |
| ACEBench | 31.59%（1023） | 37.69%（1023） | 静态、可追溯评分 |
| ToolBench Action EM | 9.65%（2258） | 16.02%（2259） | 静态评分 |
| ToolBench Primary F1 | 4.68%（2258） | 9.55%（2259） | 静态评分 |
| BFCL-v4 主短配置 | 48.54%（800/1648） | 61.04%（1006/1648） | 严格 JSON 提取；两模型均已与官方 sample_score.acc 对齐 |

ToolBench 其他指标：

- Hallucination：7.2B 13.64%，13.3B 10.98%
- Plan EM：7.2B 72.85%，13.3B 70.56%

## 验证器结果

这些结果验证接口参数和协议，不应合并为模型能力总分：

- Kimi-Verifier：两模型 inference error 均为 0/55；默认参数接受率 100%，非法参数拒绝率 20%。
- MiniMax-Verifier：tool-call 匹配率为 35.35% / 33.33%，schema 准确率为 75.00% / 72.22%（7.2B / 13.3B）。

## 未计分项目

- BFCL-v4 长上下文 memory prerequisite：曾触发至少 8193 输入 tokens 加 2048 输出 tokens，连同 2048 输出上限超过 10240，服务返回 HTTP 400；超限样本按 `context_truncated` 跳过并保留日志。
- BFCL-v4 主短配置：两模型均已完成 1648 题并生成正式报告；7.2B 为 800/1648，13.3B 为 1006/1648。追加子集遇到第三方 `format_sensitivity` 数据形状 `KeyError: 0`，属于适配器问题，按 `--ignore-errors` 跳过并保留原始日志。
- BFCL-v3 多轮：本轮正式 BFCL-v3 报告只跑非多轮 3641 题，不能把多轮空项当作 0 分。
- Tau3：需要下载 866 个知识库文件，本轮 smoke 在数据下载阶段无进展后停止，仅保留日志。
- GAIA/部分 SWE-bench：环境依赖或工具契约失败时保留原始响应和失败分类，不把环境失败改成模型分数。

## 复现与回归验证

当前 uv 依赖组 `agent` 与 `eval` 互斥，不能同时使用。Agent 专项测试使用：

```bash
uv lock --check
uv run --no-default-groups --group agent --with pytest \
  pytest -q tests/test_naive_chat.py \
    tests/test_evalscope_agent.py \
    tests/test_evalscope_agent_results.py \
    tests/test_summarize_evalscope_agent_matrix.py \
    tests/test_parallel_candidate_proxy.py
```

结果：**69 passed，2 warnings**。

报告根目录：

```text
/home/rwkv/chase/EvalScope/results/evalscope
```

BFCL-v3 正式报告：

```text
retest-bfclv3-nonmt-7p2b-20260731-timeout120-rerun1/20260731_050012/reports/.../bfcl_v3.json
retest-bfclv3-nonmt-13p3b-20260731-timeout120-rerun1/20260731_050030/reports/.../bfcl_v3.json
```

结论：本轮修复确认了请求超时、重试控制、原始响应保留、严格提取和判别链路可复现；分数仍反映模型原始 Agent 行为，没有通过格式补偿或答案补写提高成绩。正式发布门槛仍未满足，主要短板是闭环停止、失败恢复和最终答案遵循。

## BFCL-v4 主短配置验收（2026-07-31）

7.2B 运行目录：

```text
/home/rwkv/chase/EvalScope/results/evalscope/bfcl-v4-7p2b-router-full
```

官方报告为 `score=0.4854`，1648 个样本中 800 个 `acc=1`、848 个 `acc=0`。本地 acceptance report 的分类为：

```text
correct=800, model_error=848, format_invalid=0, run_errors=[]
```

两套结果完全一致，因此该 7.2B 分数可作为本轮正式可追溯结果。解析器只接受模型原始 JSON 数组；格式错误明确失败，不补写工具名、参数、结束符或答案字段。

13.3B 运行目录：

```text
/home/rwkv/chase/EvalScope/results/evalscope/bfcl-v4-13p3b-router-full
```

官方报告为 `score=0.6104`，1648 个样本中 1006 个 `acc=1`、642 个 `acc=0`。使用当前源码执行 `--report-only` 重建 acceptance report 后，分类为：

```text
correct=1006, model_error=642, format_invalid=0, run_errors=[]
```

13.3B 非聚合子集分数：`irrelevance=7.08%（240）`、`live_simple=59.30%（258）`、`multiple=80.00%（200）`、`parallel=79.00%（200）`、`parallel_multiple=65.00%（200）`、`simple_java=34.00%（100）`、`simple_javascript=38.00%（50）`、`simple_python=83.75%（400）`。主指标是报告顶层 `score=61.04%`；报告中的 `OVERALL=13.54%` 是 BFCL-v4 的另一个聚合类别，不能替代主指标。

旧安装包曾把 BFCL-v4 文件名误识别为通用 `bfcl` 并产生全量 `format_invalid`；该旧报告已被当前源码重建结果替换，不作为成绩依据。
