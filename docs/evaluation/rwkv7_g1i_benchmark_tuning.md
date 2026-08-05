# RWKV7 G1I benchmark 首批调优记录

日期：2026-08-05

## 模型与协议

- 权重：`models/rwkv7/rwkv7-g1i-1.5b-20260805-ctx16384.pth`
- 文件大小：`3055444605` bytes
- SHA-256：`32ef7b5bf4dc8bde843cf26dfad809a1f527e2e76a9e790e7d406e71bcd785da`
- 服务模型 ID：`rwkv7-g1i-1.5b`
- 服务上下文：16384 tokens；lm-eval 有效上下文为 16382 tokens
- 评测器：`lm-eval==0.4.12`，所有正式运行保留 samples 和错例分析

主入口为 `configs/eval/lm_eval.toml`，每个 selector 的独立参数位于
`configs/eval/lm_eval_benchmarks/`。本记录只覆盖已实际运行的首批 benchmark，不能
替代其余 selector 的正式结果。

## 已验证结果

| Benchmark | 样本 | 最终协议 | 结果 |
| --- | ---: | --- | ---: |
| GSM Plus | 100 | `assistant` + `fake_think`，greedy，512 tokens | flexible/strict 35.0% |
| RACE | 100 | 原生 base prompt，多选 likelihood | accuracy 32.0% |
| LongBench v2 | 20 子任务各 10 题，共 200 题 | 原生 `Answer:`，多选 likelihood | accuracy 27.5% |

这些都是本地抽样结果，不能作为完整 benchmark 分数发布。对应运行目录分别为
`.tmp/eval/lm-eval-gsm-plus-100`、`.tmp/eval/lm-eval-race-100` 和
`.tmp/eval/lm-eval-longbench2-10-raw-final`。

## GSM Plus 调优

先用固定的前 8 题比较 prompt，再在固定的前 100 题确认最终选择：

| 方案 | 样本 | Flexible | Strict | 结论 |
| --- | ---: | ---: | ---: | --- |
| `assistant` + `fake_think` | 8 | 62.5% | 62.5% | 保留候选 |
| 加“只用已知事实”等通用 system instruction | 8 | 37.5% | 25.0% | 明显退化，删除 |
| `bot` + `open_think` | 8 | 50.0% | 12.5% | 输出冗长且格式稳定性差，删除 |
| 原生 base prompt | 100 | 29.0% | 29.0% | 弱于 chat 协议 |
| `assistant` + `fake_think` | 100 | 35.0% | 35.0% | 最终配置 |

100 题中，按上游 perturbation 分类的正确率包括：digit expansion 61.5%、problem
understanding 58.3%、adding operation 38.5%、numerical substitution 38.5%、
distraction insertion 25.0%，critical thinking 0%。代表性错误不是单一格式问题：模型
会自行补出题目没有给出的“5 天”、把蛋壳当作额外消耗的鸡蛋，也会在信息不足时强行
列方程给出数值。因此没有保留针对 8 题拟合的通用 system instruction。

## RACE 诊断

100 题 accuracy 为 32.0%，68 题错误。该任务使用上游原生多选概率协议，HTTP
loglikelihood 链路返回有效分数。样本分析显示选项长度等因素可能影响概率比较，因此
没有为了抬高抽样分数而改写题目或答案协议；后续应在固定完整 split 上报告结果并单独
分析选项位置、长度与题型。

## LongBench v2 调优与修复

该 group 含 20 个四选一子任务。`limit = 10` 实际产生 200 道题和 800 个 continuation
评分请求。最初使用 chat wrapper 时暴露出 RWKV tokenizer 的边界合并：例如
`Answer:` 与 `Answer:A` 的分词并不满足简单前缀切分，旧逻辑会得到空 continuation 和
伪造的 0 logprob。适配器现改为在上下文尾部的有界窗口重新分词，既识别边界合并，又
兼容远端 tokenizer 对超长上下文的左截断。

修复后两轮 200 题比较如下：

| 协议 | Accuracy | A/B/C/D 预测次数 | 零分 continuation |
| --- | ---: | --- | ---: |
| `assistant` chat wrapper | 21.5% | 159 / 17 / 18 / 6 | 0 / 800 |
| 原生 `Answer:` | 27.5% | 78 / 69 / 14 / 39 | 0 / 800 |

原生协议同时提高分数并显著减轻 A 位置塌缩，因此 `longbench2.toml` 固定为
`profile = "none"`、`generation_prompt = "none"`。最终 800 个选项 logprob 的范围为
`[-8.7940673828125, -1.48539137840271]`，证明没有空 continuation 混入聚合。

上下文仍是主要限制。相同 200 题按 RWKV tokenizer 统计，token 数最小 10423、
中位数约 115886、P90 约 456611、最大 4835510；185/200（92.5%）超过 16382 token
并被左截断。按数据集自带长度标签，本次抽样的 short/medium/long accuracy 分别为
23.8%、23.7% 和 40.9%，但每档样本数只有 63/93/44，且后两档全部截断，不能据此
推断模型真实的完整长上下文能力。

## Qwen3.5 参考边界

[Qwen3.5-0.8B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)公开的
LongBench v2 分数为 26.1，[Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)
为 38.7。它们是 post-trained、thinking 模式和 262144 原生上下文下的官方结果；本地
RWKV 数字只是固定 200 题、16384 上下文的工程验证，不能据 27.5% 宣称胜过 0.8B。
[Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)与
[Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)模型卡没有公开同协议
LongBench v2、GSM Plus 或 RACE 表，因此不使用其他 benchmark 数字进行替代比较。

## 最终配置选择

- GSM Plus：保留 `assistant` + `fake_think`，greedy，将上限从 2048 降为 512。
- RACE：保留原生 base prompt 和上游多选 likelihood，不做抽样过拟合。
- LongBench v2：保留原生 base prompt，批大小 16，生成上限 64（该任务不生成）。
- 所有运行继续开启 `log_samples`；完整发布必须移除 smoke `limit` 并在同一 task
  version、split、上下文和 prompt 协议下比较。
