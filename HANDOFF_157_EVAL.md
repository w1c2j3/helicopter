# 157 / 8222 服务器交接与大规模 EvalScope 测评说明

更新时间：2026-08-04（Asia/Shanghai）

这份文档用于交接给后续 AI 或协作者。目标是在服务器上审计两个
`vllm-rwkv` checkout 的真实启动方式，并使用当前 Helicopter 分支进行
可追溯、可恢复的大规模 Agent 测评。不要把真实密钥、数据库密码或模型
服务令牌写入 Git、日志、Issue 或聊天记录。

## 一、当前仓库状态

- GitHub 仓库：`https://github.com/w1c2j3/helicopter`
- 分支：`updata/supported-dataset`
- 当前工作区可能包含用户保留的 `src/scoreboard-client/next-env.d.ts`
  生成文件变化；不要擅自回退。
- `results/`、`experiments/`、缓存和临时运行产物不纳入仓库。测评输出应放在
  服务器上的独立结果目录，例如：
  `/home/rwkv/chase/eval-results/<run-label>`。
- 不要修改 `rwkv-rs`、`rwkv-skills` 或服务器已有推理服务。

## 二、SSH 链路

本机 WSL 中已经配置以下 alias：

```bash
Host rwkv-8222
    HostName 47.115.88.183
    Port 8222
    User chase
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host rwkv-157
    HostName 192.168.0.157
    User rwkv
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ProxyJump rwkv-8222
```

连接验证只执行只读命令：

```bash
ssh rwkv-8222 'hostname; id -un; pwd'
ssh rwkv-157 'hostname; id -un; pwd'
```

等价完整链路是：

```bash
ssh -J chase@47.115.88.183:8222 rwkv@192.168.0.157
```

## 三、两个服务器上的代码与权重

| 主机 | `vllm-rwkv` checkout | Helicopter checkout | 权重根目录 |
|---|---|---|---|
| `rwkv-8222` | `/home/chase/vllm-rwkv` | 以独立上传目录为准 | `/home/chase/weights/BlinkDL__temp-latest-training-models/` |
| `rwkv-157` | `/home/rwkv/chase/vllm-rwkv` | `/home/rwkv/chase/helicopter` | `/home/rwkv/chase/weights/BlinkDL__temp-latest-training-models/` |

目录和服务状态必须现场确认，不能只凭端口或旧日志推断。每台服务器先执行：

```bash
hostname
git -C /home/<user>/<path>/vllm-rwkv status --short
git -C /home/<user>/<path>/vllm-rwkv log -1 --oneline
nvidia-smi
ss -ltnp
```

## 四、`vllm-rwkv` 启动审计

先阅读源码和现有启动脚本，再决定并发参数：

```bash
cd /home/<user>/<path>/vllm-rwkv
rg -n "enable-auto-tool-choice|tool-call-parser|tokenizer-mode|max-model-len|max-num-seqs|max-num-batched-tokens|VLLM_USE_V2_MODEL_RUNNER|CUDA_VISIBLE_DEVICES" .
find . -maxdepth 3 -type f \( -name '*.sh' -o -name '*.service' -o -name '*launch*' -o -name '*server*' \) -print
```

启动模板仅供核对，不得未经 GPU、权重、端口和已有进程检查直接执行：

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> \
VLLM_USE_V2_MODEL_RUNNER=1 \
/home/<user>/.venv-vllm-<verified>/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /home/<user>/weights/<verified-model>.pth \
  --served-model-name <verified-model-id> \
  --host 127.0.0.1 \
  --port <verified-port> \
  --api-key <read-from-secret-env> \
  --tokenizer-mode rwkv \
  --enable-auto-tool-choice \
  --tool-call-parser rwkv \
  --max-model-len 10240 \
  --max-num-seqs <source-derived-value> \
  --max-num-batched-tokens <source-derived-value> \
  --gpu-memory-utilization 0.95
```

必须记录但不要泄露密钥的项目：源码 commit、Python/vLLM 版本、模型绝对路径、
GPU 绑定、端口、上下文长度、批处理上限、tool-call parser 和完整启动参数。

服务确认必须使用 `/v1/models`，不能仅凭端口判断模型：

```bash
curl -fsS http://127.0.0.1:<port>/v1/models \
  -H "Authorization: Bearer ${HELICOPTER_EVAL_API_KEY}"
```

再使用一个无副作用的工具调用请求确认：`finish_reason=tool_calls`、工具名和
JSON arguments 均可被服务解析。不要通过修改模型输出或补写标签来伪造成功。

## 五、远程上传方式

上传前使用干净的 Git 归档，不上传 `.env`、`.venv`、`results/`、缓存或本地
临时文件。示例：

```bash
git archive --format=tar HEAD | gzip > /tmp/helicopter-updata-supported-dataset.tar.gz
scp /tmp/helicopter-updata-supported-dataset.tar.gz rwkv-157:/tmp/
ssh rwkv-157 'mkdir -p /home/rwkv/chase/helicopter-updata-supported-dataset && tar -xzf /tmp/helicopter-updata-supported-dataset.tar.gz -C /home/rwkv/chase/helicopter-updata-supported-dataset'
```

远程代码目录应与现有 checkout 分离，避免覆盖正在运行的任务。远程依赖统一
使用 `uv`：

```bash
cd /home/rwkv/chase/helicopter-updata-supported-dataset
uv lock --check
uv sync --no-default-groups --group agent
uv run --no-default-groups --group agent --no-sync helicopter eval evalscope --help
```

## 六、大规模测评约束

1. 先做 `/v1/models` 和单条 tool-call smoke，再做小批量回归，最后才启动全量。
2. 每个模型、端点和 benchmark 使用独立 `--work-dir`，结果放在
   `/home/rwkv/chase/eval-results/`，不要写回 Git 仓库。
3. 官方沙盒和官方判分器是最终分数来源；本地诊断只记录原始请求、模型响应、
   提取结果、判别依据、错误分类和性能指标。
4. 保持项目原有 system prompt、工具定义、消息语义和顺序；naive Chat 或
   parallel-candidate 只能在请求发送边界做格式/路由适配。
5. 不补写模型没有生成的 tool call、结束标记、选项、字段或答案；上下文超限时
   记录并跳过该样本，不能伪装为正确。
6. 7.2B 和 13.3B 使用不同端口、不同结果目录；未经明确授权不要启动、停止或
   重启另一模型，也不要占用其他 GPU。
7. 每轮保存命令、配置、源码 commit、服务信息、原始输出和官方报告；先保留
   原始结果，再修复工程问题并回归。

推荐的 native Agent 调用形态：

```bash
uv run --no-default-groups --group agent --no-sync helicopter eval evalscope \
  --config configs/example.toml \
  <model-alias> <dataset> \
  --model-catalog configs/models/g1h-single-replica.toml \
  --base-url http://127.0.0.1:<verified-port>/v1 \
  --api-key "${HELICOPTER_EVAL_API_KEY}" \
  --no-server \
  --strategy function_calling \
  --agent-environment local \
  --work-dir /home/rwkv/chase/eval-results/<run-label>
```

长工具目录或证据上下文需要 `parallel-candidate` 时，先确认当前分支已有实现
和回归测试，再单独记录路由配置；不要把它当成模型服务参数。

## 七、密钥、数据库和 judge

服务器秘密配置只从以下文件读取，不打印内容：

```text
/home/rwkv/chase/helicopter/.env.remote
/home/rwkv/chase/helicopter/.env
```

可能需要的变量名包括：

```text
SCOREBOARD_DB_PASSWORD
HELICOPTER_EVAL_API_KEY
HELICOPTER_JUDGE_API_KEY
OPENAI_API_KEY
```

项目数据库与 `chase_rwkv_skills` 分离，连接参数以 `.env.remote` 和服务器当前
配置为准。需要 judge 的 benchmark 必须单独标记为外部服务依赖，不能把 judge
失败混入模型错误或答案提取失败。

## 八、故障处理与交接记录

- 先记录 `git status`、服务 PID、GPU、命令和 work directory，再做任何操作。
- 不使用 `git reset --hard`、`git checkout --` 或覆盖式同步来清理别人的修改。
- 不停止未知 PID；停止任务前必须核对完整命令行确实属于当前 benchmark。
- 工程错误可修复并加回归测试；模型行为问题只记录，不放宽判分器。
- 结论必须区分：接口错误、上下文超限、格式/提取失败、判别器错误和模型回答错误。
- 新 AI 接手时先读本文件、`README.md`、当前分支 `git log -3`，再检查两台服务器
  的源码 commit 与服务状态；不要直接启动全量测评。
