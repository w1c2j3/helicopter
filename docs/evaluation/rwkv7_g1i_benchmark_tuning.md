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

## 多选 Wave A

固定 `limit = 20` 实测 MuSR、LogiQA2、WMDP、BBQ 和 ToxiGen。Qwen3.5 官方模型卡
没有公开这些同名、同协议结果，因此这里只做协议和 prompt A/B，不借用其他任务分数
作为目标。

| Benchmark | 原生 prompt | `assistant` prompt | 最终选择 |
| --- | ---: | ---: | --- |
| MuSR | acc_norm 33.3%（60 题） | 26.7% | 原生 |
| LogiQA2 | acc 30.0%，acc_norm 30.0% | 20.0%，25.0% | 原生 |
| WMDP | acc 28.3%（60 题） | 25.0% | 原生 |
| ToxiGen | acc/acc_norm 50.0% | 60.0%/50.0% | 原生；normalized 无改善 |
| BBQ likelihood | acc 45.0% | 不适用 | 保留上游原生协议并注明偏差 |

MuSR 的 60 题来自 murder mysteries、object placements、team allocation 各 20 题；
所有 continuation 都有有效 logprob。错例分析现按 `acc_norm` 的字符长度归一化规则恢复
模型选择，不再把 likelihood-only 样本误报为空输出。LogiQA2 需要 Hugging Face 自定义
loader；审阅确认 loader 只下载官方 GitHub JSONL 并解析字段后，在该 selector 独立设置
`trust_remote_dataset_code = true`，运行结束立即恢复默认拒绝状态。

ToxiGen 的原生与 chat 协议都在 20 题中 18 次预测 `Yes`，说明 50% normalized accuracy
掩盖了明显标签偏置，不能只看聚合分数。WMDP 三个 leaf 分别为 bio 25%、chem 30%、
cyber 30%，当前抽样接近四选一随机水平，chat wrapper 没有改善。

上游 BBQ likelihood task 为每题加入 12 个 continuation：两个实体答案加十种
`Unknown` 同义表达。20 题中模型从未选择 unknown，ambiguous accuracy 0%、
disambiguated accuracy 90%；总分 45% 同时受 continuation 长度影响。额外测试上游
`bbq_generate`：`assistant + fake_think` 为 10%，去掉 thinking 并加入“信息不足回答
Unknown”指令后为 35%，但 ambiguous accuracy 仍为 0%。因此没有用较低的生成分数
替换正式 selector，也不把 45% 解释为无偏的知识/推理能力。

## 多语种多选 smoke

固定每个 leaf `limit = 5` 验证 Belebele、XNLI 和 XCOPA 的完整 group 展开、官方数据集
加载与 RWKV likelihood 链路。结果分别写入
`.tmp/eval/lm-eval-multilingual-choice-5` 和
`.tmp/eval/lm-eval-xnli-xcopa-5`：

| Benchmark | 语言/leaf | 样本 | 结果 |
| --- | ---: | ---: | ---: |
| Belebele | 122 | 610 | acc/acc_norm 32.13% |
| XNLI | 15 | 75 | acc 44.0% |
| XCOPA | 11 | 55 | acc 40.0% |

XNLI 和 XCOPA 的上游 lm-eval YAML 仍引用 Hugging Face 已废弃的无 namespace 数据集
别名。独立 benchmark 配置把它们分别映射回同一官方数据集 `facebook/xnli` 和
`cambridgeltl/xcopa`；评测器通过 lm-eval 0.4.12 的 task factory 构造完整 group，
没有改写 task 的 prompt、split、指标或语言列表。所有 335 个候选 continuation 都返回
有效 logprob。

XCOPA 两个选项的预测次数为 27/28，没有明显位置塌缩。XNLI 的三个选项预测次数为
43/9/23，而本次固定前五题在 15 种语言重复后的标签次数为 15/30/30，存在明显的首选项
偏置。由于每种语言只有五题，逐语言分数的标准误很大；这组数字只证明协议可运行并暴露
偏置，不能作为正式多语种能力排名。此前 MuSR、LogiQA2、WMDP 和 LongBench v2 的
固定样本 A/B 均显示 chat wrapper 不改善或显著退化 likelihood 结果，因此这里保留上游
原生 base prompt，避免在 5 题/语言上继续过拟合。

## 生成 Wave A

固定每个 leaf `limit = 5` 实测 DROP、XQuAD 和 `mgsm_cot_native`。XQuAD 展开 12 个
语言 leaf，MGSM 的固定 tag 同时展开 11 个 English-CoT 与 11 个 native-CoT leaf；加上
DROP 共 175 个样本、35 个 leaf。最终保留的正式入口为
`configs/eval/lm_eval_generation_wave_a_smoke.toml`，结果在
`.tmp/eval/lm-eval-generation-wave-a-5`。

| Benchmark | `assistant + fake_think` | 额外格式指令 | 原生 prompt（最终） |
| --- | ---: | ---: | ---: |
| DROP | EM 0%，F1 17.2% | 未测 | EM/F1 20.0% |
| XQuAD | EM 11.7%，F1 13.1% | 未测 | EM 43.3%，F1 50.4% |
| MGSM | flexible 35.5%，strict 0% | 25.5%，5.5% | 42.7%，22.7% |

DROP 与 XQuAD 的上游 prompt 已包含问答结构；chat wrapper 会让模型输出一个孤立的
`>`、解释性整句或与原文不一致的答案。相同固定样本切回原生 causal prompt 后，XQuAD
更常直接续写答案 span，例如英文前三题由 1/3 exact 提高到 2/3 exact。XQuAD 上游 YAML
仍引用无 namespace 的 `xquad`，独立配置只将数据集恢复为同一官方
`google/xquad`；其 SQuAD EM/F1 实现依赖 `transformers.data.metrics.squad_metrics`，
因此 lm-eval 独立环境现显式锁定兼容的 `transformers`。

MGSM 的 strict filter 要求输出包含 `The answer is <number>`。chat wrapper 常以
`\\boxed{}` 结束，导致 flexible 能抽取数字而 strict 全部失配；强制格式的 system
instruction 虽修复少量 strict 样本，却使 flexible 下降 10 个百分点。原生 prompt 的
few-shot 示例能更自然地诱导目标短语，并同时提高两项指标，因此最终不保留额外指令。
原生 MGSM 在单独试跑时为 flexible/strict 40.0%/20.0%，在三任务正式重跑时为
42.7%/22.7%；两次均为 greedy 且样本相同，这个约 2–3 点差异应视为服务并发或数值
非确定性，而不是能力变化。

Qwen3.5-0.8B/2B 的官方模型卡没有公开同名、同 lm-eval 协议的 DROP、XQuAD 或 MGSM
结果，因此本轮不使用近似任务数字代替基线。CruxEval 会在本机执行模型生成的 Python；
上游 reliability guard 明确不是安全沙箱，所以未把它混入本轮普通评测，后续需要在独立
OS/container 沙箱中运行。

## 语言建模与语法 smoke

`configs/eval/lm_eval_language_modeling_smoke.toml` 固定 `limit = 10` 运行 WikiText、
Pile-10k、LAMBADA OpenAI 和 BLiMP。困惑度与 likelihood 任务必须保留原生 causal
prompt；套 chat template 会改变语料条件概率，不属于可比较的 prompt 调优。

| Benchmark | 样本 | 结果 |
| --- | ---: | ---: |
| WikiText | 10 文档 | word PPL 11.756，byte PPL 1.599，BPB 0.678 |
| Pile-10k | 10 文档 | word PPL 26.785，byte PPL 1.666，BPB 0.737 |
| LAMBADA OpenAI | 10 题 | acc 50.0%，PPL 5.356 |
| BLiMP | 67 leaf 各 10 题，共 670 题 | acc 82.09% |

这些都是前缀抽样，尤其 10 文档 PPL 会强烈受文档长度和领域影响，不能替代完整 split。
BLiMP 最低的 leaf 包括 matrix-question NPI licensor 20%、existential-there
quantifiers-2 30%、wh-vs-that-with-gap 30% 和 principle-A reconstruction 40%；相对地，
多个名词一致性、被动语态和照应一致性 leaf 在该十题切片为 100%。这里保留 leaf 级
样本和错例，不把 82.09% 单一聚合值解释为所有语法现象都稳定。

Paloma 的 16 个语料 leaf 另用 `configs/eval/lm_eval_paloma_smoke.toml` 尝试加载，但
官方 `allenai/paloma` 当前为 gated dataset，环境没有授权，因而明确标记为 blocked；
没有绕过门禁或用其他困惑度语料替代。Qwen3.5-0.8B/2B 官方模型卡也没有这些同名、
同 lm-eval 协议结果，本轮不构造伪基线。

## 翻译与开放生成状态

`configs/eval/lm_eval_wmt14_smoke.toml` 在 WMT14 English-to-French 固定前五题得到
BLEU 13.34、chrF 39.59、TER 79.28（TER 越低越好）。样本显示前三题基本保持法语并
覆盖原意，第四题使用可接受的近义词；第五题则完全偏离原句，生成了一条英文汽车产业
新闻。只有五题时 corpus BLEU 的 bootstrap 标准误为 8.31，因此这组数字仅用于检查
生成与 corpus metric 链路，不能与完整 3003 题结果直接比较。

FLORES 与 WMT14 已拆成独立 smoke 入口，避免 gated 数据集失败使先完成的 selector
无法落盘。`configs/eval/lm_eval_flores_smoke.toml` 对应的官方 `facebook/flores` 当前
需要授权，状态为 blocked；没有改用其他 FLORES 镜像或相似翻译集。RealToxicityPrompts
的上游指标必须把生成文本发送给 Perspective API，当前没有 `PERSPECTIVE_API_KEY`，
因此同样保持 blocked，没有执行无指标的生成后伪称 benchmark 已完成。

## 最终配置选择

- GSM Plus：保留 `assistant` + `fake_think`，greedy，将上限从 2048 降为 512。
- RACE：保留原生 base prompt 和上游多选 likelihood，不做抽样过拟合。
- LongBench v2：保留原生 base prompt，批大小 16，生成上限 64（该任务不生成）。
- Belebele、XNLI、XCOPA：保留原生 base prompt；XNLI/XCOPA 仅修复到官方 namespace，
  不改变上游任务协议。
- DROP、XQuAD、MGSM：三者均切回原生 base prompt；XQuAD 仅修复到官方 namespace。
- WikiText、Pile-10k、LAMBADA、BLiMP：固定原生 base prompt；Paloma 保持 gated blocked。
- WMT14：保留原生翻译 prompt；FLORES 与 RealToxicityPrompts 保持明确 blocked。
- 所有运行继续开启 `log_samples`；完整发布必须移除 smoke `limit` 并在同一 task
  version、split、上下文和 prompt 协议下比较。
