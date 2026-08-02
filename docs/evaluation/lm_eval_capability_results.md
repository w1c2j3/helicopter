# lm-eval 能力补充套件结果

评测日期：2026-08-02。本文记录 `configs/eval/lm_eval_capabilities.toml` 的完整
evaluation split 结果。该套件与 `configs/eval/lighteval.toml` 的 selector 清单严格
不相交，覆盖阅读理解、翻译、语言建模、句法语言学和长上下文检索。

当前提交先记录已完成的 1.5B RWKV 与 Qwen3.5-0.8B Base 下界对照；
Qwen3.5-2B Base 上界对照正在按同一协议运行，完成后追加到同一报告。

## 对齐协议

| 项目 | 固定值 |
| --- | --- |
| evaluator | `lm-eval==0.4.12` |
| task selectors | `race`、`wmt14-en-fr`、`lambada_openai`、`blimp`、`longbench_passage_retrieval_en` |
| prompt | base-model 原生 prompt，不使用 chat template |
| context | `max_length=16382` |
| decoding | task 原生确定性配置 |
| dataset | 完整 evaluation split，`limit=None` |
| samples | 每个 task 都启用逐样本落盘 |
| offline | 模型和数据准备完成后启用三个 Hugging Face offline 开关 |

RWKV 使用本地 `rwkv-vllm` HTTP 后端、FP16 WKV；Qwen 使用 lm-eval 原生 `hf`
后端、BF16。不同 batch size 只用于适配 8GB 显存，不改变 prompt、输出或指标语义。
WMT 的 BLEU、chrF 和 TER 都保留 lm-eval 默认 100 次 bootstrap；TER 越低越好。

## 已完成结果

“RWKV 优势”按指标方向统一为正数表示 RWKV 更好：高优指标为
`RWKV - Qwen`，低优指标为 `Qwen - RWKV`。括号内为 stderr。

| 能力 / 指标 | 方向 | 样本数 | RWKV7 1.5B | Qwen3.5-0.8B Base | RWKV 优势 |
| --- | :---: | ---: | ---: | ---: | ---: |
| RACE accuracy | ↑ | 1,045 | 0.363636 (0.014888) | 0.355024 (0.014810) | +0.008612 |
| WMT14 BLEU | ↑ | 3,003 | 17.202313 (0.387524) | 14.989362 (0.375330) | +2.212952 |
| WMT14 chrF | ↑ | 3,003 | 40.641350 (0.434106) | 39.576008 (0.422961) | +1.065343 |
| WMT14 TER | ↓ | 3,003 | 78.743663 (0.854241) | 85.782195 (1.310768) | +7.038532 |
| LAMBADA accuracy | ↑ | 5,153 | 0.662721 (0.006587) | 0.507665 (0.006965) | +0.155055 |
| LAMBADA perplexity | ↓ | 5,153 | 4.844395 (0.110354) | 11.664836 (0.357653) | +6.820441 |
| BLiMP group accuracy | ↑ | 67,000 | 0.816463 (0.001338) | 0.811642 (0.001352) | +0.004821 |
| LongBench retrieval score | ↑ | 200 | 0.045000 (0.014695) | 0.045000 (0.014695) | 0.000000 |

当前下界对照中，RWKV 在阅读理解、翻译、语言建模和 BLiMP 上均领先 0.8B
Base；LongBench passage retrieval 两者同为 4.5%，是这组结果最明显的共同短板。
不同参数量之间只报告逐指标差异，不计算跨任务总分。

## 模型与产物身份

| 模型 | 精确身份 | 运行后端 |
| --- | --- | --- |
| RWKV7 1.5B | `rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth`，SHA-256 `22fe129988f6e98480b344075597259a13ae4201c1d8dedf987246772e613586` | 本地 RWKV-vLLM，`rwkv7-g1i-1.5b`，FP16 WKV |
| Qwen3.5-0.8B Base | revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | lm-eval `hf`，BF16 |

RWKV 标准产物位于
`.tmp/eval/lm-eval-capabilities-20260802/rwkv-full/`，其中：

- `results.json`: `150f4481eb25a89ed31fb3408da6ea2ccb96693495be5da78393a98a138eda63`
- `summary.json`: `41b45c3cc4805c4c81458ec9e7976bdb6c495c6a077f6e84129cf35376075c27`
- `artifacts.json`: `c4e12ec23004d67ed96b3c5c9863ca9e0ed3c55ba2d5718f5561a6360270af6c`

Qwen3.5-0.8B 的五个 lm-eval 原生结果文件位于
`.tmp/eval/lm-eval-capabilities-20260802/qwen08-full/`，SHA-256 分别为：

- RACE: `7f50e67a6fb5930978621549db7ad4ebb909fdc5463aa02462f4d2956434f4f0`
- WMT14: `5021eaef87395f34f860c4bfef3d8bebb4376289f2db753fa8f784576479f33d`
- LAMBADA: `2af51993678f618b0c144403ac343bf316c754bd829eec8f3d2279d13600be54`
- BLiMP: `6189fe795adcd78f13371569cd5e116a69f512f2ca82b38736fda890d6fb8391`
- LongBench: `5f3d6d9ec3d4d9edb1bf5794a1b92001da2500d71ef032520dda85f2b1ebc794`

两种模型都落盘 76,401 条 sample 记录。RWKV 的标准产物为权限 `0600`；运行目录
位于 `.tmp/`，不会进入 Git。

## 运行说明

Hugging Face 主站不可达时，数据和 Qwen snapshot 通过 `hf-mirror.com` 准备；正式
评分全程使用本地 cache 和 offline 模式。Qwen3.5-0.8B 最初将 RACE、LAMBADA、
BLiMP 合并为 batch 32 时，RACE 的长输入和 248K 词表 logits 导致 OOM。该次尝试
没有生成结果文件，也不计入分数；正式运行按 task 输入形状拆分为 RACE 1、LAMBADA
8、BLiMP 32、WMT14 32、LongBench 1，并全部退出码为 0。
