# lm-eval-harness 评估

Helicopter 将 `lm-eval==0.4.12` 作为独立评估器运行，支持两条明确的执行路径：

- RWKV 模型通过已经运行的 RWKV-vLLM HTTP pool 评估，并生成逐 benchmark、逐题和
  错例产物，也可发布到项目 Scoreboard。
- Hugging Face 等非 RWKV 模型直接交给原生 lm-eval-harness，模型后端、task、指标和
  输出格式均保持上游语义。

公共入口统一为：

```bash
.venv/bin/helicopter eval --evaluator lm-eval --config <config.toml>
```

`--evaluator` 不能省略；Helicopter 的默认评估器仍是 LightEval。

## 选择执行路径

入口只根据 TOML 顶层字段路由，不根据模型名称猜测：

| 配置 | 执行路径 | 适用模型 | 产物 |
| --- | --- | --- | --- |
| `backend = "vllm_http"` | Helicopter RWKV 适配器 | RWKV-vLLM HTTP 服务 | lm-eval 原始结果、项目摘要、完整逐题记录、错例报告、可选 Scoreboard 发布 |
| 不写 `backend` | 原生 lm-eval | `hf`、`vllm`、API 等 lm-eval 原生后端 | lm-eval 上游输出 |

不要为原生模型写 `backend = "hf"`。除 `vllm_http` 外的 `backend` 值会被拒绝；原生
配置也不能混入 `prompt`、`benchmark_configs`、`pool_manifests`、`weights` 或
`wkv_modes` 等 RWKV 专属字段。

## RWKV 五分钟上手

### 1. 准备环境

本地启动 RWKV-vLLM 需要 Linux 或 WSL2、Python 3.12、NVIDIA GPU、可用的 CUDA
toolkit（`nvcc` 必须在 `PATH` 或 `CUDA_HOME/bin`）以及 RWKV `.pth` 权重。

首次拉取仓库后初始化 submodule，并只安装本地评估所需组件：

```bash
git submodule update --init --recursive

INSTALL_COMPONENTS=vllm-rwkv,lm-eval,dev \
  ./scripts/install_local.sh
```

缺少 Ubuntu 基础编译工具时，可在安装命令前增加 `INSTALL_SYSTEM_DEPS=1`；该选项会
通过 `sudo apt-get` 安装基础工具，但不会安装 NVIDIA driver 或 CUDA toolkit。

安装器会创建两个环境：

- `.venv`：Helicopter CLI 和 RWKV-vLLM 运行时。
- `.venv-lm-eval`：固定的 lm-eval 评估环境。

本地服务脚本默认从 `.venv` 启动 vLLM；自定义环境可通过
`HELICOPTER_VLLM_VENV=/absolute/path/to/venv` 覆盖。旧 checkout 中已有的
`src/infer/vllm-rwkv/.venv-rwkv` 仍会被兼容发现。

如果只连接已经部署好的 RWKV-vLLM pool，或只运行原生 lm-eval，可使用更小的安装：

```bash
INSTALL_COMPONENTS=lm-eval,dev ./scripts/install_local.sh
```

不要把 LightEval 和 lm-eval 手工装入同一个虚拟环境；两者的 `datasets` 版本要求不同。

### 2. 启动 RWKV-vLLM

建议先在终端 A 启动服务。仓库安装的默认 checkpoint 是
`models/rwkv7/rwkv7-g1i-1.5b-20260805-ctx16384.pth`，可直接使用：

```bash
./scripts/run_rwkv_vllm.sh
```

启动脚本会输出并原子写入
`.tmp/runtime/rwkv-vllm-pool.json`。该 manifest 包含 endpoint、并发、上下文长度、
WKV mode、权重名称和 SHA-256；评估入口会自动发现它。

使用其他实际存在的 checkpoint 时，再设置绝对路径；`--max-model-len` 必须与其上下文
能力一致：

```bash
export RWKV_MODEL_PATH=/absolute/path/to/rwkv-model.pth
./scripts/run_rwkv_vllm.sh --max-model-len 16384
```

当前本地启动脚本按项目验证环境设置 `/usr/local/cuda-13.0`。使用其他 CUDA 布局、
远程服务或多个 replica 时，请自行启动兼容服务并使用后文的 HTTP pool manifest。

### 3. 预检并运行本地评分验证

终端 B 使用仓库自带的本地评分配置：

```bash
./scripts/run_lm_eval.sh \
  configs/eval/lm_eval_local_quick.toml \
  --dry-run

./scripts/run_lm_eval.sh \
  configs/eval/lm_eval_local_quick.toml
```

`--dry-run` 会验证 TOML、固定 lm-eval 版本、task selector、live HTTP pool、model id 和
有效上下文，但不会加载数据集或执行评分。如果当前没有健康的本地服务且没有显式
manifest，`run_lm_eval.sh` 会先启动 RWKV-vLLM；它只回收自己启动的服务，日志写入
`.tmp/runtime/rwkv-vllm.log`。

该配置在 WikiText、LAMBADA、RACE 和 GSM-Plus 上各运行 50 条（约 200 条），覆盖
rolling likelihood、choice scoring 和 generation；它用于本地诊断出分，不能替代完整
可发布评测。完成后直接查看指标和错题：

```bash
jq '.metrics' .tmp/eval/lm-eval-local-quick/summary.json

jq '.benchmark_artifacts[] | {task_name, samples, errors, report_path}' \
  .tmp/eval/lm-eval-local-quick/artifacts.json

less .tmp/eval/lm-eval-local-quick/benchmarks/race/report.md
```

这三步完成了安装、服务预检、真实评分和逐题产物检查。只验证安装和服务连通性时，
可改用 `configs/eval/lm_eval_quickstart.toml` 的 5 条 ARC Easy；原有的完整 28 selector
套件仍保持在 `configs/eval/lm_eval.toml`。

## 原生 lm-eval 模型

不写 `backend` 时，TOML 会完整交给 `python -m lm_eval run --config ...`。例如创建一份
Hugging Face 配置：

```toml
model = "hf"
model_args = { pretrained = "EleutherAI/pythia-70m", dtype = "float32" }
tasks = ["hellaswag"]
device = "cpu"
batch_size = 1
limit = 5
output_path = ".tmp/eval/lm-eval-native"
log_samples = true
```

然后使用同一个入口：

```bash
./scripts/run_lm_eval.sh configs/eval/my_native_lm_eval.toml --dry-run
./scripts/run_lm_eval.sh configs/eval/my_native_lm_eval.toml
```

原生 dry-run 解析上游配置和 task，不加载模型。`run_lm_eval.sh` 识别到原生路由后不会
启动 RWKV-vLLM。该路径使用 lm-eval 原生 result tracker，不生成 Helicopter 的
`summary.json`、逐 benchmark 错例目录或 Scoreboard campaign；这些增强只属于显式的
RWKV `vllm_http` 路径。

原生字段和后端参数以固定版本的 lm-eval CLI 为准：

```bash
.venv-lm-eval/bin/python -m lm_eval run --help
```

## RWKV 配置

最小的 RWKV 配置如下：

```toml
schema_version = 1
backend = "vllm_http"

tasks = ["arc_easy", "hellaswag", "gsm8k"]
output_dir = ".tmp/eval/my-run"
batch_size = 16
eot_token_id = 0
max_gen_toks = 512
log_samples = true

[prompt]
profile = "none"

[generation_kwargs]
do_sample = false
```

顶层字段说明：

| 字段 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `schema_version` | 是 | 无 | 固定为 `1` |
| `backend` | 是 | 无 | RWKV 路径固定为 `vllm_http` |
| `tasks` | 是 | 无 | lm-eval task、group、tag 或 glob selector；每项必须至少匹配一个 task |
| `output_dir` | 是 | 无 | 输出目录；相对路径按项目根目录解析 |
| `batch_size` | 否 | `1` | 一次交给 HTTP backend 的请求批量；实际并发仍受 manifest 限制 |
| `eot_token_id` | 否 | `0` | 空上下文和首个 rolling window 使用的 RWKV 条件 token |
| `max_gen_toks` | 否 | `256` | task 未声明生成长度时的默认值 |
| `limit` | 否 | 不限制 | 每个 leaf task 的 smoke 样本上限；发布配置禁止设置 |
| `log_samples` | 否 | `true` | 保留原始 samples 并生成逐题与错例产物 |
| `benchmark_configs` | 否 | 空 | 每个 selector 的独立配置文件，必须与 `tasks` 一一对应 |
| `task_include_paths` | 否 | 空 | 自定义 lm-eval task 目录，相对当前 TOML 所在目录解析 |
| `publish` | 否 | `false` | 是否在评估过程中发布 Scoreboard campaign |
| `weights`、`wkv_modes`、`pool_manifests` | 发布时 | 从单 manifest 推导 | 定义生产 weight/WKV mode 执行矩阵 |

`output_dir`、`pool_manifests` 和 `weights` 等字符串支持完整值形式的环境变量引用，
例如 `"${LM_EVAL_FP16_POOL_MANIFEST}"`；不支持在一个字符串中拼接多个变量。

### 查找和校验 task

task 名称、group、tag、few-shot、filter 和 metric 都由 lm-eval 0.4.12 提供，项目不维护
另一份名字映射：

```bash
.venv-lm-eval/bin/python -m lm_eval ls tasks
.venv-lm-eval/bin/python -m lm_eval ls groups
.venv-lm-eval/bin/python -m lm_eval ls tags
.venv-lm-eval/bin/python -m lm_eval validate --tasks arc_easy,gsm8k
```

`limit` 应只用于链路验证。selector 是 group 或 tag 时，limit 会作用于每个展开后的
leaf task，因此总题数可能大于 limit。正式可比较结果必须删除 `limit`。

### 逐 benchmark 配置

不同 benchmark 需要不同 prompt、生成长度或安全开关时，在总配置中引用独立 TOML：

```toml
tasks = ["race", "gsm_plus"]
benchmark_configs = [
  "lm_eval_benchmarks/race.toml",
  "lm_eval_benchmarks/gsm_plus.toml",
]
```

每份独立配置至少包含：

```toml
schema_version = 1
selector = "gsm_plus"
batch_size = 8
max_gen_toks = 512

[prompt]
profile = "assistant"
generation_prompt = "none"
fewshot_as_multiturn = false

[generation_kwargs]
do_sample = false
```

外部配置会继承顶层值，并可覆盖 `batch_size`、`max_gen_toks`、`limit`、`prompt` 和
`generation_kwargs`。加载器要求它们与 `tasks` 中的 selector 完全一一对应，并拒绝
缺失、重复、额外 selector 或多个 selector 展开到同一 leaf task。

只有经过审阅且确实需要时，才在对应 benchmark 文件中启用：

- `trust_remote_dataset_code = true`：允许该数据集的 Hugging Face 自定义 loader。
- `confirm_run_unsafe_code = true`：允许运行 lm-eval 标记为 unsafe 的 task。
- `dataset_path_override = "namespace/canonical-name"`：修正上游已失效的无 namespace
  数据集别名，必须仍是同一官方数据集。
- `[dataset_kwargs_override]`：修正同一数据集的 schema 或 data files，不得替换成相似
  benchmark。

这些开关默认关闭，只在该 selector 加载期间生效，并记录到结果摘要。

### Prompt

```toml
[prompt]
profile = "assistant" # none | bot | assistant | function_calling
generation_prompt = "none" # none | open_think | fake_think
system_instruction = "Return only the requested answer."
num_fewshot = 2
fewshot_as_multiturn = true
```

- `none` 是默认值，保留 lm-eval 原生 base-model prompt。做跨模型或官方 harness 对比时
  优先使用它。
- `bot`、`assistant` 和 `function_calling` 启用仓库维护的 RWKV renderer，并自动加入
  对应 turn stop。
- `assistant` 的生成起始格式为 `User: <题目>\nAssistant:`，默认不在
  `Assistant:` 后预填 thinking 内容。
- `open_think` 预填 `<think`，`fake_think` 预填 `<think></think`；两者会改变评测协议，
  不能与 `none` 的结果直接混算。
- 最终生效的 profile SHA-256、system instruction、few-shot 和 generation prompt 会写入
  `summary.json` 与发布 payload。

WikiText 等标准 PPL 任务必须使用 `profile = "none"`，否则测到的是套入 chat prompt 后
的文本，而不是标准语料困惑度。需要修改 task 自身的 `doc_to_text`、description 或
few-shot 样例时，应维护自定义 YAML 并通过 `task_include_paths` 加载，不要编辑
`.venv-lm-eval/site-packages/lm_eval/tasks`。

### 生成参数

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

还支持 `until`、`max_gen_toks`、`max_new_tokens` 和 `ignore_eos`。
`max_gen_toks` 与 `max_new_tokens` 不能同时设置；当前 HTTP backend 只支持
`num_beams = 1`。`do_sample = false` 会发送 `temperature = 1.0, top_k = 1`，语义仍是
逐 token argmax，同时兼容不接受 `temperature = 0` 的 RWKV rapid sampler。

## vLLM HTTP pool

本地 `run_rwkv_vllm.sh` 自动生成的 manifest 无需手写。连接远程或多 replica 服务时，
在私有 `.env.local` 中设置绝对路径：

```dotenv
HELICOPTER_VLLM_POOL_MANIFEST=/run/helicopter/rwkv-vllm-pool.json
```

`.env.local` 必须是当前用户拥有的普通文件，权限为 `0600`，且不能是 symlink：

```bash
chmod 600 .env.local
```

manifest schema 与 LightEval HTTP backend 相同：

```json
{
  "schema_version": 1,
  "global_step": 0,
  "wkv_mode": "fp16",
  "vllm_version": "0.23.1.dev0",
  "max_model_len": 16384,
  "weight_sha256": "<64 lowercase hex>",
  "weight_display_name": "rwkv-model.pth",
  "replicas": [
    {
      "base_url": "http://127.0.0.1:8000",
      "max_concurrency": 16
    }
  ]
}
```

启动预检会访问每个 replica 的 `/health` 和 `/v1/models`，要求 endpoint 全部可用且只
服务同一个 model id。评分使用 `/tokenize` 与 `/v1/completions`，支持 lm-eval 的
`loglikelihood`、`loglikelihood_rolling` 和 `generate_until` 三种标准请求。

RWKV-vLLM 在 decoder prompt 和零生成评分请求中各保留一个内部位置，因此评估器的
有效上下文为 manifest `max_model_len - 2`，配置的所有 `max_gen_toks` 必须小于该值。
HTTP client 不继承 shell proxy；429、5xx 和传输错误最多切换一个 replica 重试，其他
4xx 直接失败。

## 结果与错例

RWKV 路径始终写出 `results.json`、`summary.json` 和 `artifacts.json`。当
`log_samples = true`（默认值）时，每个执行单元还包含完整分析产物：

```text
<output_dir>/
├── results.json
├── summary.json
├── artifacts.json
├── error_analysis.json
├── bad_cases.json
├── error_analysis.md
└── benchmarks/
    └── <task>/
        ├── summary.json
        ├── records.jsonl
        ├── errors.jsonl
        └── report.md
```

- `results.json`：lm-eval `simple_evaluate` 的完整可序列化结果；开启 sample 日志时也
  包含原始 samples。
- `summary.json`：model id、权重摘要、WKV mode、上下文、prompt、生成参数、task 版本和
  metrics。
- `artifacts.json`：所有逐 benchmark 产物的稳定索引。
- `records.jsonl`：全部逐题记录，保留 `doc_id`、问题、原始输出、模型/标准答案、判定
  metric、选项分数和所有 filter 结果。
- `errors.jsonl`：全部可判定错误和生成质量异常，不做抽样。
- `report.md`：可直接阅读的 benchmark 摘要和前 20 条错例。

RWKV runner 不会把所有 benchmark 的原始 samples 同时保留在内存中。每次
`simple_evaluate` 返回后，runner 会在 `output_dir` 同一文件系统的私有 staging 目录中
立即写入该批原始 samples、逐题记录和错例产物，然后释放该批 Python 对象；全部批次完成
后再流式组装兼容的 `results.json` 并替换正式产物。运行失败时 staging 会自动删除，已有
完整结果保持不变。最终组装期间磁盘会短暂同时保存 sample spool 和正式结果，因此
`output_dir` 所在盘仍需预留至少一份完整 `results.json` 的额外空间。

常用检查命令：

```bash
OUTPUT=.tmp/eval/lm-eval-quickstart

jq '.metrics' "$OUTPUT/summary.json"
jq '.benchmark_artifacts[] | {task_name, samples, errors, report_path}' \
  "$OUTPUT/artifacts.json"
jq -c '{doc_id, question, model_output, model_answer, standard_answer, why_wrong}' \
  "$OUTPUT/benchmarks/arc_easy/errors.jsonl" | less
```

多 filter task 会按 `doc_id` 合并后统计，原始输出和每个 filter verdict 都会保留。
BLEU、chrF、TER 等连续指标不会被伪造为二元“答对/答错”；报告只单独标记重复、元回答
或极低参考重合等质量异常。纯 likelihood task 没有自由生成文本时不会伪造
`model_answer`，而是保留目标 continuation 和选项分数。

已有 `results.json` 可补生成分析而无需重跑模型：

```bash
.venv-lm-eval/bin/python -m helicopter_lm_eval.analysis \
  --results .tmp/eval/old-run/results.json \
  --examples-per-task 5
```

后处理写入 `analysis_artifacts.json` 并记录源文件 SHA-256，不修改历史
`artifacts.json`。重复使用同一 `output_dir` 开始新评估时，runner 会清理自己管理的旧
分析目录，避免不同运行的题目混在一起。

## 发布到 Scoreboard

### 在线生产 campaign

`configs/eval/lm_eval_campaign.toml` 与 LightEval 的生产 campaign 约束对齐：必须提供
权重身份、`fp16` 和 `fp32io16` 两种 WKV mode，以及每个矩阵单元对应的 live pool。

将私有值写入 `.env.local`：

```dotenv
WEIGHT_PATH=/absolute/path/to/weights
LM_EVAL_CAMPAIGN_WEIGHT=model.pth
LM_EVAL_FP16_POOL_MANIFEST=/run/helicopter/lm-eval-fp16.json
LM_EVAL_FP32IO16_POOL_MANIFEST=/run/helicopter/lm-eval-fp32io16.json
HELICOPTER_SCOREBOARD_URL=https://scoreboard.example
HELICOPTER_SCOREBOARD_TOKEN=replace-with-private-token
```

```bash
chmod 600 .env.local

.venv/bin/helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval_campaign.toml \
  --dry-run

.venv/bin/helicopter eval \
  --evaluator lm-eval \
  --config configs/eval/lm_eval_campaign.toml
```

manifest 中的 `weight_sha256`、`weight_display_name` 和 `wkv_mode` 必须与矩阵单元完全
一致。dry-run 会预检全部 pool 和 Scoreboard 合同但不创建 campaign。正式运行先创建
`lm-eval-campaign-v1`，逐 leaf task 发布 `lm-eval-task-v1`，全部预期 task 到齐后才
finalize。`publish = true` 时任何顶层或 benchmark `limit` 都会在配置阶段被拒绝。

### 发布已有产物

已有 RWKV lm-eval 产物可以独立发布，不重跑模型，也不修改原文件：

```bash
.venv/bin/helicopter publish \
  --evaluator lm-eval \
  --output-dir .tmp/eval/completed-run \
  --dry-run

.venv/bin/helicopter publish \
  --evaluator lm-eval \
  --output-dir .tmp/eval/completed-run
```

该入口验证 evaluator 版本、权重身份、WKV mode、metrics、samples、`doc_id` 覆盖和
`artifacts.json` 引用，再预检 Scoreboard。可重复传入 `--output-dir` 发布同一 task
矩阵的多个 weight/mode 单元；不同 task 集合必须分开发布。

旧产物缺少权重摘要时，可显式提供来源可信的身份：

```bash
.venv/bin/helicopter publish \
  --evaluator lm-eval \
  --output-dir .tmp/eval/completed-run \
  --weight-sha256 '<64 lowercase hex>' \
  --weight-display-name model.pth \
  --dry-run
```

显式值与产物已有摘要冲突时会失败。历史运行时版本未知时会记录为
`not-recorded-in-artifact`，不会拿当前环境版本冒充历史版本。

## 与 LightEval 的关系

lm-eval 不是 LightEval 的替代入口，两者共享生产约束，但各自保留上游语义：

| 能力 | LightEval | lm-eval RWKV 路径 |
| --- | --- | --- |
| 公共入口 | `helicopter eval --evaluator lighteval` | `helicopter eval --evaluator lm-eval` |
| 隔离环境 | `.venv-lighteval` | `.venv-lm-eval` |
| task/metric 所有者 | LightEval `Pipeline` | lm-eval `TaskManager` / `simple_evaluate` |
| RWKV 服务 | 同一 HTTP pool manifest 和 preflight | 同一 HTTP pool manifest 和 preflight |
| 本地详细结果 | results JSON、details Parquet | `results.json`、完整 `records.jsonl`、`errors.jsonl` 和报告 |
| Scoreboard | LightEval campaign 合同 | `lm-eval-campaign-v1` / `lm-eval-task-v1` |
| 抽样结果发布 | 禁止 | 禁止 |

`configs/eval/lm_eval_capabilities.toml` 和默认 lm-eval 清单用于补充 LightEval 清单未覆盖
的能力。正式比较必须固定同一个 evaluator、selector、task version、数据 revision、
few-shot、prompt、生成参数、上下文和完整 split；不能直接把 LightEval 与 lm-eval 的
同名但不同协议分数相减。

## 仓库预设

| 配置 | 用途 | 是否限制样本 |
| --- | --- | --- |
| `configs/eval/lm_eval_quickstart.toml` | 5 道 ARC Easy，验证安装、服务和错例产物 | 是 |
| `configs/eval/lm_eval_local_quick.toml` | 本地快速评分：WikiText、LAMBADA、RACE、GSM-Plus 各 50 样本 | 是 |
| `configs/eval/lm_eval.toml` | 默认完整 RWKV 社区 benchmark 套件，含逐 benchmark profile | 否 |
| `configs/eval/lm_eval_ppl.toml` | 标准 WikiText PPL，禁用 chat prompt | 否 |
| `configs/eval/lm_eval_capabilities.toml` | 与 LightEval 默认清单不重叠的能力补充 | 否 |
| `configs/eval/lm_eval_catalog_delta.toml` | 经固定 lm-eval 注册表过滤后的用户清单增量 | 否 |
| `configs/eval/lm_eval_qwen35.toml` | 固定 Qwen3.5 对齐 selector 的本地协议比较 | 否 |
| `configs/eval/lm_eval_campaign.toml` | 双 WKV mode 的生产发布矩阵 | 否 |

`configs/eval/lm_eval_benchmarks/` 保存默认套件的逐 benchmark 配置；文件中的 prompt、
安全开关和数据覆盖都是评测协议的一部分。新 G1I 权重的实测与调参记录见
[`rwkv7_g1i_benchmark_tuning.md`](rwkv7_g1i_benchmark_tuning.md)，能力套件结果见
[`lm_eval_capability_results.md`](lm_eval_capability_results.md)，代表性错例分析见
[`lm_eval_bad_case_analysis.md`](lm_eval_bad_case_analysis.md)。

## 常见问题

### 找不到 `helicopter` 或 lm-eval Python

重新运行对应安装组件，并优先使用仓库内的固定入口 `.venv/bin/helicopter`。不要依赖
当前 shell 是否已 activate 虚拟环境。

### `RWKV model not found`

设置实际 checkpoint：

```bash
export RWKV_MODEL_PATH=/absolute/path/to/model.pth
```

### `vLLM replica preflight failed` 或连接被拒绝

确认服务仍在运行，检查 `.tmp/runtime/rwkv-vllm.log`，并验证 manifest endpoint：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models | jq .
```

使用远程服务时，确认 `HELICOPTER_VLLM_POOL_MANIFEST` 是绝对路径且所有 replica 可从
评估机直接访问。评估 HTTP client 不使用 shell proxy。

### task 不存在

使用固定环境的 `lm_eval ls` 查名称。不要用相似 benchmark、临时自建 YAML 或另一套
数据集冒充缺失 task；升级 lm-eval 后必须重新审阅 task 和数据协议。

### 数据集下载失败或要求授权

网络问题可以通过 `HF_ENDPOINT` 或预下载缓存解决。gated 数据集必须先在官方页面接受
条款，并通过 shell 或权限为 `0600` 的 `.env.local` 提供只读 `HF_TOKEN`；镜像不能
绕过授权。没有授权时应记录为 blocked，不能替换数据集后宣称完成同一 benchmark。

### `max_gen_toks` 超过上下文

适配器的有效长度是 manifest `max_model_len - 2`。降低生成长度，或用模型真实支持的
更大 `max_model_len` 重新启动服务；不要只改 manifest 而不改服务。

### 发布配置拒绝 `limit`

这是预期的完整性保护。用 quickstart/smoke 配置验证链路，正式运行删除所有 limit 后
重新 dry-run，不能发布抽样分数。
