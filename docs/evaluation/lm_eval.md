# lm-eval-harness RWKV-vLLM 后端

该适配器使用已经运行的 RWKV-vLLM HTTP pool，不会在评估进程中加载权重或创建
第二份 vLLM engine。它实现 lm-eval 模型后端的三个标准请求：
`loglikelihood`、`loglikelihood_rolling` 和 `generate_until`。因此可以直接运行
HellaSwag、ARC、MMLU、TruthfulQA、GSM8K、IFEval、WikiText 等常见 harness task。

开箱即用的本地入口只有一条命令：

```bash
./scripts/run_lm_eval.sh
```

它会自动复用健康的本地服务；没有服务时会启动 RWKV-vLLM、等待就绪、执行评测，
最后回收自己启动的服务。服务日志写入 `.tmp/runtime/rwkv-vllm.log`。要运行另一份
配置，只需把 TOML 路径作为第一个参数：

```bash
./scripts/run_lm_eval.sh configs/eval/lm_eval_ppl.toml
```

本地运行不需要手写 endpoint、pool manifest 或 `.env.local`。首次运行或改完配置后，
可在命令末尾加 `--dry-run` 做配置、任务和服务预检。已有固定服务时也可直接执行
`.venv/bin/helicopter eval --evaluator lm-eval --config configs/eval/lm_eval.toml`。

`lm-eval` 固定为 `0.4.12`，安装在独立的 `.venv-lm-eval`。可通过
`HELICOPTER_LM_EVAL_PYTHON` 覆盖解释器路径。LightEval 仍是
`helicopter eval` 的默认 evaluator，现有 MaxRL 验证不受影响。

## 配置

```toml
schema_version = 1
backend = "vllm_http"

tasks = ["arc_easy", "hellaswag", "mmlu", "gsm8k", "ifeval"]
output_dir = ".tmp/eval/lm-eval"
batch_size = 64
eot_token_id = 0
max_gen_toks = 2048
log_samples = true

[prompt]
profile = "bot"
generation_prompt = "open_think"
fewshot_as_multiturn = true

[generation_kwargs]
do_sample = false
```

- `tasks` 接受 lm-eval task、group、tag 和 glob pattern；每个 selector 必须至少
  匹配一项。task 定义、few-shot、过滤器和指标仍完全由 lm-eval 提供。
- `output_dir` 可为绝对路径，或相对执行命令时的项目目录；环境变量引用使用
  `${VARIABLE}` 形式。
- `batch_size` 控制一次提交给 HTTP pool 的评分窗口数量，pool manifest 中每个
  replica 的 `max_concurrency` 继续作为实际并发上限；默认示例使用 `64`。
- `eot_token_id` 默认是 RWKV 的 `0`，用于为空上下文和首个 rolling window 提供
  条件 token。
- `max_gen_toks` 是 task 未指定生成长度时的默认值。task 自己的
  `max_gen_toks` 或 `max_new_tokens` 优先；提示词会从左侧截断，为输出保留空间。
- `limit` 是可选的正整数，只用于本地 smoke test；`publish = true` 时配置解析会
  直接拒绝 `limit`，防止将抽样结果发布成完整评测。
- `log_samples` 默认开启；原始 lm-eval sample 信息会进入结果文件并生成错例分析。
  只有 WikiText/Pile 等没有逐题二元答案的纯 PPL 诊断才应显式关闭。
- `task_include_paths` 可选，加载仓库维护的自定义 lm-eval task 目录；相对路径以当前
  TOML 所在目录为基准。标准对标配置不得借此静默替换上游同名 task。

### 逐 benchmark 配置

总入口通过 `benchmark_configs` 引用每个 benchmark 的独立 TOML：

```toml
tasks = ["wikitext", "gsm_plus"]
benchmark_configs = [
  "lm_eval_benchmarks/wikitext.toml",
  "lm_eval_benchmarks/gsm_plus.toml",
]
```

每个文件必须声明唯一 `selector`，并可覆盖批大小、生成长度、smoke limit、prompt
和生成参数。只有经过审阅且确实依赖 Hugging Face 自定义 loader 的 benchmark 才可
显式设置 `trust_remote_dataset_code = true`；它只在该 selector 加载期间生效并立即
恢复。lm-eval 自身标记为 unsafe 的 task 则使用独立的
`confirm_run_unsafe_code = true`。两个开关默认均为 `false` 并写入结果摘要：

若上游 task 仍使用 Hugging Face 已废弃的无 namespace 别名，可设置
`dataset_path_override = "namespace/canonical-dataset"`。覆盖只作用于该 selector
解析出的 leaf task，运行结束后恢复；它必须指向同一官方数据集，不能用相似数据替代。
上游 YAML 若只因固定 `datasets==3.6.0` 的特征 schema 变化而无法加载，可用
`[dataset_kwargs_override]` 替换该 selector 的 `dataset_kwargs`；必须保持相同数据、
split 和字段语义，并在独立 benchmark 文件中记录完整 schema。

```toml
schema_version = 1
selector = "gsm_plus"
batch_size = 8
max_gen_toks = 512
confirm_run_unsafe_code = false
trust_remote_dataset_code = false
dataset_path_override = "namespace/canonical-dataset"

[prompt]
profile = "assistant"
generation_prompt = "fake_think"
fewshot_as_multiturn = false

[generation_kwargs]
do_sample = false
```

加载器要求 `tasks` 与外部配置中的 selector 一一对应，并拒绝缺失、额外、重复或
解析到同一 task 的配置。运行时每个 selector 单独调用 lm-eval，完成后再无损合并
原生 metrics、group、sample 和 task config；`summary.json` 与 Scoreboard 会保留
每个 benchmark 最终生效的配置。默认总入口的独立文件位于
`configs/eval/lm_eval_benchmarks/`。

新 G1I 权重的首批实测、prompt A/B、错例与 LongBench v2 上下文限制记录在
`docs/evaluation/rwkv7_g1i_benchmark_tuning.md`。

### RWKV prompt

项目在 `src/eval/lm_eval/prompts.py` 中维护与 RWKV-vLLM 一致的 prompt renderer。
TOML 的 `[prompt]` 表控制实际协议：

```toml
[prompt]
# none 保留 lm-eval 原生 base-model prompt；其余值启用 RWKV chat renderer。
profile = "bot" # none | bot | assistant | function_calling
generation_prompt = "open_think" # none | open_think | fake_think
system_instruction = "Follow the requested answer format."
num_fewshot = 2
fewshot_as_multiturn = true
```

- `none` 是 schema 默认值，不加 chat template，适合与官方 lm-eval causal LM
  baseline 对比。
- `bot`、`assistant` 和 `function_calling` 使用仓库维护的 RWKV 角色格式，并自动把
  对应的用户回合边界加入 generation stop。
- `open_think` 在回答前写入 `<think`；`fake_think` 写入 `<think></think`；`none`
  只写 assistant 起始标记。概率型多选和生成式推理对 thinking prefix 的需求不同，
  不应在同一个正式结果中混用后再与标准 baseline 直接比较。
- `system_instruction`、最终生效的 profile SHA-256、few-shot 和 generation prompt
  都写入 `summary.json` 与 Scoreboard sampling config，保证结果可回溯。
- 上游明确固定为 0-shot 的 task 不会被 `num_fewshot` 强制覆盖；这是 lm-eval 的原生
  安全行为。

需要修改某个 benchmark 自身的 `doc_to_text`、`description` 或 few-shot 样例时，
在仓库中维护自定义 task YAML，并通过配置加载：

```toml
task_include_paths = ["../../tasks/lm_eval"]
tasks = ["rwkv_gsm8k"]
```

不要直接编辑 `.venv-lm-eval/site-packages/lm_eval/tasks`；环境重建会丢失修改，也无法
从结果中证明实际使用了哪份 prompt。

### 生成参数

`[generation_kwargs]` 会通过 lm-eval 原生 `gen_kwargs` 覆盖所有生成类 task 的 YAML
参数，评分型 task 不受影响：

```toml
[generation_kwargs]
do_sample = true
temperature = 0.96
top_p = 0.76
top_k = 32
min_p = 0.0
presence_penalty = 1.0
frequency_penalty = 0.1
repetition_penalty = 1.0
penalty_decay = 0.988
seed = 1234
```

生成请求支持 harness 常用的 `until`、`max_gen_toks`、`max_new_tokens`、
`do_sample`、`temperature`、`top_p`、`top_k`、`min_p`、`seed` 以及 RWKV penalty
参数。当前后端只支持
单路生成，task 若要求 `num_beams > 1` 会明确失败，而不是静默改变评估语义。
`do_sample = false` 会编码为 `temperature = 1.0, top_k = 1`：它仍是逐 token
argmax，但同时兼容 RWKV-vLLM 的 rapid sampler（该 sampler 不接受
`temperature = 0`）。

## HTTP 协议

本地使用 `./scripts/run_rwkv_vllm.sh` 时无需配置本节内容。启动器会根据实际的
host、port、上下文长度、并发、模型 SHA-256 和 WKV mode 自动生成
`.tmp/runtime/rwkv-vllm-pool.json`，`helicopter eval` 在未显式指定清单时自动使用它。

训练期、远程服务或多 replica 部署仍应在私有 `.env.local` 或 `.env.remote` 中提供
受控清单的绝对路径：

```dotenv
HELICOPTER_VLLM_POOL_MANIFEST=/run/helicopter/vllm-pool.json
```

manifest schema 与 LightEval HTTP backend 相同。启动时会访问每个 replica 的
`/health` 和 `/v1/models`，确保所有 endpoint 可用并且只服务同一个 model id。
显式的 `HELICOPTER_VLLM_POOL_MANIFEST` 始终优先于本地自动发现路径。

评估使用以下接口：

1. `/tokenize` 对 harness 生成的原始 prompt、context 和 continuation 分词，固定
   `add_special_tokens = false`；RWKV endpoint 返回的首个 EOT/BOS token 会被移除。
2. `loglikelihood` 和 `loglikelihood_rolling` 通过 `/v1/completions` 接收 token
   IDs，并设置 `echo = true`、`max_tokens = 0`、
   `logprobs = 1`、`prompt_logprobs = 1` 和 `return_token_ids = true`。
3. `generate_until` 同样使用 `/v1/completions`，传入左截断后的 prompt token
   IDs、task 的 stop sequences 和采样参数；服务端和适配器都会执行 stop 截断。
4. 适配器验证评分返回 token 完全一致，忽略首 token 的 `null` logprob，只累加
   continuation 对应位置的自然对数概率。后续 rolling window 不以 EOT 开头时，
   RWKV-vLLM 自动补入的 EOT 会在返回结果中被正规化，避免 token 和 logprob 错位。

RWKV tokenizer 可能在 context/continuation 边界合并 token，且远端 tokenizer 会对
超长输入左截断。适配器会在上下文尾部的有界窗口重新分词后切分 continuation；不要
使用 `len(tokenize(context))` 直接切整段 token，否则多选项可能被误判为空 continuation。

HTTP client 不继承 shell proxy，避免本机或内网服务被全局代理转发。HTTP 429、
5xx 和传输错误最多切换一个 replica 重试；其他 4xx 直接失败。

RWKV-vLLM 会为 decoder prompt 内部增加一个 token，并且对
`echo + max_tokens=0` 的请求仍保留至少一个生成位置。因此适配器暴露的有效长度为
manifest `max_model_len - 2`，rolling window 再保留一个条件 token。

`prompt.profile = "none"` 保持 lm-eval base-model 语义：harness 构造的 prompt
不会被适配器再次套一层 chat template。这一点对基于 continuation 概率的多选指标
很重要，也使结果能与 lm-eval 的其他 causal LM 后端直接比较。通用
`configs/eval/lm_eval.toml` 展示 RWKV `bot` 调优协议；Qwen、capability、catalog
delta、PPL 和生产对标配置显式锁定 `none`，避免历史协议漂移。
WikiText 等 PPL task 不得套 RWKV chat prompt，否则指标不再是标准语料困惑度；应使用
`configs/eval/lm_eval_ppl.toml` 单独运行。

## Qwen3.5 对齐套件

`configs/eval/lm_eval_qwen35.toml` 将任务名对齐到 Qwen3.5 模型卡公开的语言评测
表：`MMLU-Pro`、`MMLU-Redux`、`C-Eval`、`GPQA`、`IFEval` 和 `MMMLU`。
固定 selector 如下：

| Qwen 表中名称 | lm-eval 0.4.12 selector | 本地协议 |
| --- | --- | --- |
| MMLU-Pro | `mmlu_pro` | 5-shot、确定性生成、exact match |
| MMLU-Redux | `mmlu_redux_generative` | 0-shot、确定性生成、exact match |
| C-Eval | `ceval-valid` | validation split、多选概率 |
| GPQA | `gpqa_diamond_cot_zeroshot` | Diamond、0-shot CoT、exact match |
| IFEval | `ifeval` | 0-shot、确定性生成、严格/宽松指令指标 |
| MMMLU | `mmmlu` | 14 种语言、0-shot、多选概率 |

运行 RWKV 对齐套件：

```bash
helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval_qwen35.toml \
  --dry-run

helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval_qwen35.toml
```

Qwen3.5 官方没有 1.5B 型号。最接近的官方尺寸是
[`Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) 和
[`Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base)，因此 1.5B RWKV
应以 0.8B/2B 作为参数量下界和上界，不能标记为“同参数量”。精确 1.5B 对照只能
选择旧代模型，例如 Qwen2.5-1.5B，而不能再称为 Qwen3.5 对齐。

截至 2026-08-01，Qwen 官方模型卡给出的后训练模型参考分如下：

| 模型与模式 | MMLU-Pro | MMLU-Redux | C-Eval | GPQA | IFEval | MMMLU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B non-thinking | 29.7 | 48.5 | 46.4 | 未报告 | 52.1 | 34.1 |
| Qwen3.5-0.8B thinking | 42.3 | 59.5 | 50.5 | 11.9 | 44.0 | 44.3 |
| Qwen3.5-2B non-thinking | 55.3 | 69.2 | 65.2 | 未报告 | 61.2 | 56.9 |
| Qwen3.5-2B thinking | 66.5 | 79.6 | 73.2 | 51.6 | 78.6 | 63.1 |

来源：[`Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) 和
[`Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) 官方模型卡。这些分数使用
后训练模型、chat template、thinking/non-thinking 模式及官方生成参数；Qwen 没有
公开足以逐项复现整张表的完整 harness 配置。因此它们只能作为外部参考，不能与
本项目默认 base-model 结果直接相减。

可比较的本地结果必须同时固定：模型类型（优先 Base）、`lm-eval==0.4.12`、上述
selector、task version、few-shot、无 chat template、确定性解码、数据 revision、
`limit = null` 和主指标。尤其不能把默认套件的普通 `mmlu` 分数填入 Qwen 表的
`MMLU-Pro` 或 `MMLU-Redux` 列。`SuperGPQA` 也不是 `GPQA Diamond`，当前配置不
用后者冒充前者。

## 输出

成功运行后，`output_dir` 中包含：

- `results.json`：lm-eval `simple_evaluate` 的完整可序列化结果。
- `summary.json`：稳定的项目级摘要，包含 lm-eval 版本、model id、global step、
  WKV mode、上下文长度、prompt profile、生成参数、task 版本和 metrics。
- `error_analysis.json`：按 task 与 task family 汇总可判定样本的正确/错误数、错误率、
  错误类型和生成质量诊断。
- `bad_cases.json`：确定性抽取的代表性错例，包含 `task_name + doc_id`、模型答案、
  标准答案、选择题各选项分数和错误 margin，可回查原始 sample。
- `error_analysis.md`：有界的人类可读错例报告，每个 task family 最多展示三例，
  防止大型 group 的叶子任务淹没其他能力问题。
- `benchmarks/<task>/report.md`：每个实际执行 task 的独立摘要与前 20 条错例，直接打开
  即可查看问题、模型答案、标准答案和判错原因。
- `benchmarks/<task>/records.jsonl`：该 task 的全部逐题记录；每行包含 `doc_id`、
  `status`、`model_answer`、`standard_answer`、判定 metric、选项分数和 prompt 摘要。
- `benchmarks/<task>/errors.jsonl`：该 task 的全部错误和生成质量异常，不做抽样。

默认配置仍只写本地，不创建 Scoreboard campaign。每个执行单元会写出
`results.json`、`summary.json`、`artifacts.json`，启用 `log_samples` 时还会按 task
写入 `samples/*.json`，并自动生成上述三份错误分析产物。连续生成指标（如 BLEU、
chrF、TER）不会被强行转成二元“答错”；报告只将重复循环、元回答和极低参考重合
列为 quality diagnostics。loglikelihood task 没有自由生成答案时，报告会保留目标词
及其分数并明确将 `model_answer` 置空，不伪造模型回答。WikiText-only 运行仍可使用
`configs/eval/lm_eval_ppl.toml`。

评测完成后最常用的查看方式：

```bash
# 看有哪些 benchmark 日志及错误数
jq '.benchmark_artifacts[] | {task_name, samples, errors, report_path}' \
  .tmp/eval/lm-eval/artifacts.json

# 直接看某个 benchmark 的人类可读报告
less .tmp/eval/lm-eval/benchmarks/gsm8k/report.md

# 筛选某个 benchmark 的全部错误：模型答案、标准答案、判错原因
jq -c '{doc_id, question, model_answer, standard_answer, why_wrong}' \
  .tmp/eval/lm-eval/benchmarks/gsm8k/errors.jsonl | less
```

已有 `results.json` 可单独补生成分析，无需重跑模型：

```bash
.venv-lm-eval/bin/python -m helicopter_lm_eval.analysis \
  --results .tmp/eval/lm-eval-capabilities/results.json \
  --examples-per-task 5
```

后处理会另写 `analysis_artifacts.json` 并记录源 `results.json` 的 SHA-256，不修改历史
`artifacts.json`，避免破坏已经发布的产物摘要；同样会补齐每个 task 的
`benchmarks/<task>/` 日志目录。

## 生产 campaign

`configs/eval/lm_eval_campaign.toml` 提供与 LightEval campaign 对齐的发布模式：

- `weights` 中的路径必须位于 `WEIGHT_PATH` 下，禁止绝对路径、`..` 和 symlink；
  runner 会计算每份权重的 SHA-256，并拒绝内容重复的权重。
- `wkv_modes` 在发布模式必须同时包含 `fp16` 和 `fp32io16`。
- `pool_manifests` 按 weight-major、随后按 `wkv_modes` 顺序一一对应执行矩阵。
  每份 manifest 必须声明与权重文件一致的 `weight_sha256`、
  `weight_display_name` 和对应 WKV mode。
- group 和 tag 在 campaign 合同中递归展开为叶子 task，但执行时仍由 lm-eval
  原生 selector 负责加载和评分。
- dry-run 会预检全部 pool 和 Scoreboard 合同，并输出完整 execution units 与
  expected task 数量；不会创建 campaign。
- 正式运行先创建 `lm-eval-campaign-v1`，逐矩阵单元运行并发布
  `lm-eval-task-v1`，全部 task 到齐后才 finalize。重复内容使用 canonical digest
  幂等写入，冲突内容失败关闭。

生产 manifest 在原有字段外必须包含：

```json
{
  "weight_sha256": "<64 lowercase hex>",
  "weight_display_name": "model.pth"
}
```

发布还要求 `HELICOPTER_SCOREBOARD_URL` 和
`HELICOPTER_SCOREBOARD_TOKEN`。Scoreboard 同时接受原有 LightEval v3/v2 合同和
lm-eval v1 合同，历史 LightEval API 保持兼容。

```bash
export WEIGHT_PATH=/absolute/path/to/weights
export LM_EVAL_CAMPAIGN_WEIGHT=model.pth
export LM_EVAL_FP16_POOL_MANIFEST=/run/helicopter/lm-eval-fp16.json
export LM_EVAL_FP32IO16_POOL_MANIFEST=/run/helicopter/lm-eval-fp32io16.json
export HELICOPTER_SCOREBOARD_URL=https://scoreboard.example
export HELICOPTER_SCOREBOARD_TOKEN=replace-with-private-token

helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval_campaign.toml \
  --dry-run
```

## 安装

```bash
INSTALL_COMPONENTS=lm-eval,dev scripts/install_local.sh
```

`lm-eval` 依赖组固定安装 `lm-eval[ifeval,longbench]==0.4.12` 和
`datasets==3.6.0`，并安装与其他运行时一致的 `transformers`。前者把 IFEval
规则校验依赖（`langdetect`、`immutabledict` 和
`nltk`）以及 LongBench 的 `jieba`、`fuzzywuzzy`、`rouge` 带入独立环境；后者保留
CMMLU 等上游 dataset-script 任务所需的加载能力（`datasets>=4` 已移除该能力）。
XQuAD 的上游指标实现仍从 `transformers.data.metrics.squad_metrics` 导入 SQuAD
归一化与 F1 逻辑，因此该依赖不能从 lm-eval 隔离环境中裁掉。
部署时不应改为只安装基础 `lm-eval` 包，也不能放宽 datasets 上界。

## 能力补充套件

`configs/eval/lm_eval_capabilities.toml` 只选择默认 LightEval 清单没有覆盖的任务，
避免重复消耗算力：

| 维度 | lm-eval selector | 协议与主指标 |
| --- | --- | --- |
| 阅读理解 | `race` | high-school split，multiple-choice accuracy |
| 翻译 | `wmt14-en-fr` | English to French，BLEU/TER/chrF |
| 语言建模 | `lambada_openai` | last-word accuracy/perplexity |
| 句法语言学 | `blimp` | 67 个最小对组任务，group accuracy |
| 大海捞针类检索 | `longbench_passage_retrieval_en` | 30 段 Wikipedia 中定位目标摘要，retrieval score |

该套件固定 `lm-eval==0.4.12`、无 chat template、完整 evaluation split、
`limit = null`、确定性生成，并保留 sample artifacts。Qwen3.5 对标必须使用同一组
selector、相同 task revision 和相同最大输入长度；本机 1.5B RWKV 对应的官方
Qwen3.5 参数量包络仍是 0.8B Base 与 2B Base，而不是不存在的 1.5B 型号。

### 与 LightEval 去重

去重的唯一基准是 `configs/eval/lighteval.toml` 中的既有评测清单。能力补充套件
不会再次运行以下 selector：

| LightEval 维度 | 已纳入 LightEval 清单的 selector |
| --- | --- |
| 知识与常识 | `mmlu`、`mmlu_pro`、`mmlu_redux_2`、`mmlu_sr_question_answer`、`gpqa:diamond`、`gpqa:main`、`arc:challenge`、`arc:easy`、`hellaswag`、`bigbench_hard`、`agieval`、`truthfulqa:mc`、`winogrande`、`openbookqa`、`commonsenseqa`、`ceval_zho_mcf`、`kmmlu`、`med_qa`、`med_mcqa` |
| 数学与推理 | `gsm8k`、`math_500`、`aime24`、`aime25`、`olympiad_bench`、`minerva_math`、`svamp`、`beyond_aime`、`brumo25`、`hmmt_feb_2025`、`math_odyssey`、`comp_math_24_25`、`gaokao_2023_english`、`answer_judge`、`simpleqa_verified` |
| 代码 | `humaneval`、`humaneval_cn`、`humaneval_fix`、`humaneval_plus`、`mbpp`、`mbpp_plus`、`lcb:codegeneration` |
| 指令遵循 | `ifeval`、`ifbench_test`、`ifbench_multiturn` |

这里的“已纳入”表示 selector 属于 LightEval campaign 合同；实际执行时，固定
LightEval 版本无法解析的 selector 会被明确记录为 `skipped`，不能记作成功出分。
仓库测试会解析两份 TOML 并断言 selector 集合不相交，防止后续维护时重新引入
重复任务。

### 用户清单增量

`configs/eval/lm_eval_catalog_delta.toml` 是用户补充清单经过两层过滤后的精确增量：
先去除上表中已经归 LightEval 管理的项目，再去除固定 `lm-eval==0.4.12` 任务注册表
不存在的项目。此前完成的能力补充套件（RACE、WMT14、LAMBADA、BLiMP、LongBench
检索）与这份补充清单没有重叠。

| 处理结果 | 用户清单项目 | lm-eval selector / 原因 |
| --- | --- | --- |
| 本轮正式评测 | GPQA-Extended | `gpqa_extended_zeroshot`，仅 Extended，0-shot multiple-choice `acc/acc_norm` |
| 本轮正式评测 | CMMLU | `cmmlu`，67 个学科，0-shot multiple-choice，按样本量聚合 `acc/acc_norm` |
| 不属于固定 lm-eval | AMC23 | `lm-eval==0.4.12` 注册表和任务源码均无该任务 |
| 不属于固定 lm-eval | SWE-bench Verified、Multilingual、Pro | `lm-eval==0.4.12` 注册表和任务源码均无这些任务 |

因此本轮 campaign 必须且只能解析为 `gpqa_extended_zeroshot` 与 `cmmlu`。不得用
自建 YAML、外部 runner 或名称近似任务填补缺失项；若以后升级 lm-eval，需要单独
评审协议和数据版本，不能静默改变本次结果合同。GPQA 数据集在 Hugging Face 上有
访问条款，正式运行前必须接受条款并提供只读 token；镜像只能解决网络问题，不能
绕过授权。

WMT14 同时报告 BLEU、chrF 和 TER。BLEU/chrF 越高越好，TER 越低越好；
lm-eval 0.4.12 的上游 WMT YAML 未声明 TER 的方向，因此运行时会出现默认方向警告，
但不会改变 TER 数值。报告和对比必须按“越低越好”解释 TER。

2026-08-02 的 RWKV7 1.5B、Qwen3.5-0.8B Base 和 Qwen3.5-2B Base 完整实测、
运行协议、stderr、样本数及产物 SHA-256 见
[`lm_eval_capability_results.md`](lm_eval_capability_results.md)；对应机器可读清单为
[`lm_eval_capability_results.json`](lm_eval_capability_results.json)。
RWKV 错题、跨任务错误模式、LongBench 协议风险与复测门槛见
[`lm_eval_bad_case_analysis.md`](lm_eval_bad_case_analysis.md)。

用户清单过滤后的原生 lm-eval 增量执行状态与 CMMLU 完整对标结果见
[`lm_eval_catalog_delta_results.md`](lm_eval_catalog_delta_results.md)；对应机器可读合同为
[`lm_eval_catalog_delta_results.json`](lm_eval_catalog_delta_results.json)。GPQA-Extended
在没有已授权只读 Hugging Face token 时明确记为 `blocked`，不会记作测评成功。

`--dry-run` 会校验配置、selector、manifest、所有 replica 和 lm-eval 版本，并输出
解析后的 task/group 类型及 output type，但不会下载数据集或执行评分。
