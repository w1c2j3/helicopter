"use client";

import { useEffect, useMemo, useState } from "react";

import { getAdminToken, setAdminToken } from "../lib/admin_token";
import { api } from "../lib/api";
import type { AdminEvalOptionsResponse } from "../lib/dtos/api/admin/eval/options";
import type { AdminEvalStatusResponse } from "../lib/dtos/api/admin/eval/status";

const TERMINAL = new Set(["idle", "completed", "cancelled", "failed"]);
const FALLBACK_MODELS = ["g1h-1.5b", "g1h-2.9b", "g1h-7.2b", "g1h-13.3b"];
const FALLBACK_TASKS = ["aime_2024", "aime_2025", "math_500", "gsm8k_test", "mmlu_test", "ifeval"];

function statusClass(status: string): string {
  if (status === "running" || status === "starting" || status === "cancelling") return "stat-run";
  if (status === "paused") return "stat-warn";
  if (status === "completed") return "stat-good";
  if (status === "failed") return "stat-bad";
  return "stat-idle";
}

export function AdminPage() {
  const [options, setOptions] = useState<AdminEvalOptionsResponse | null>(null);
  const [status, setStatus] = useState<AdminEvalStatusResponse | null>(null);
  const [draft, setDraft] = useState<Record<string, unknown>>({});
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const load = () => {
    Promise.all([api.adminOptions(), api.adminDraft(), api.adminStatus()])
      .then(([nextOptions, nextDraft, nextStatus]) => {
        setOptions(nextOptions);
        setDraft(nextDraft);
        setStatus(nextStatus);
        setConnectionError(null);
      })
      .catch((error: unknown) => {
        setConnectionError(error instanceof Error ? error.message : String(error));
      });
  };

  useEffect(() => {
    load();
    const timer = window.setInterval(() => {
      api.adminStatus().then(setStatus).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, []);

  const running = !!status && !TERMINAL.has(status.status);

  return (
    <div className="task-admin-page">
      <section className="task-admin-intro">
        <div>
          <strong>评测任务管理</strong>
          <span>本地 vllm-rwkv · 单模型权重部署 · 多 benchmark 批量评测</span>
        </div>
        <TokenField onChange={load} />
      </section>

      {connectionError ? (
        <div className="error-bar">
          管理后端暂不可用：{connectionError}。页面仍可配置任务，但连接恢复前不会启动真实评测。
        </div>
      ) : null}

      <div className="task-admin-layout">
        <TaskComposer
          connected={!connectionError && options !== null}
          disabled={running}
          draft={draft}
          options={options}
          onStatus={setStatus}
        />
        <RunMonitor status={status} onStatus={setStatus} />
      </div>
    </div>
  );
}

function TaskComposer({
  connected,
  disabled,
  draft,
  options,
  onStatus,
}: {
  connected: boolean;
  disabled: boolean;
  draft: Record<string, unknown>;
  options: AdminEvalOptionsResponse | null;
  onStatus: (status: AdminEvalStatusResponse) => void;
}) {
  const configs = options?.configs ?? [];
  const [config, setConfig] = useState("");
  const [model, setModel] = useState("");
  const [tasks, setTasks] = useState<string[]>([]);
  const [gpuIds, setGpuIds] = useState<string[]>([]);
  const [gpuText, setGpuText] = useState("");
  const [search, setSearch] = useState("");
  const [parallel, setParallel] = useState("1");
  const [maxRetries, setMaxRetries] = useState("0");
  const [rerun, setRerun] = useState(false);
  const [dryRun, setDryRun] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  useEffect(() => {
    const draftModels = Array.isArray(draft.models) ? draft.models.map(String) : [];
    const draftTasks = Array.isArray(draft.tasks) ? draft.tasks.map(String) : [];
    setConfig(String(draft.config || configs[0] || ""));
    setModel(draftModels[0] || options?.model_select?.[0] || FALLBACK_MODELS[0]);
    setTasks(draftTasks.length ? draftTasks : (options?.domains ?? FALLBACK_TASKS).slice(0, 1));
    setGpuText(String(draft.gpus || ""));
    setParallel(String(draft.parallel || "1"));
    setMaxRetries(String(draft.max_retries || "0"));
    setRerun(Boolean(draft.rerun));
    setDryRun(Boolean(draft.dry_run));
  }, [configs, draft, options]);

  const availableModels = useMemo(() => {
    const details = options?.model_options ?? [];
    const inConfig = details.filter((item) => !config || item.configs.includes(config));
    return (inConfig.length ? inConfig : details).map((item) => item.name);
  }, [config, options?.model_options]);
  const modelNames = availableModels.length ? availableModels : (options?.model_select ?? FALLBACK_MODELS);
  const taskNames = options?.domains?.length ? options.domains : FALLBACK_TASKS;
  const visibleTasks = taskNames.filter((task) => task.toLowerCase().includes(search.trim().toLowerCase()));
  const modelDetail = options?.model_options?.find((item) => item.name === model);
  const detectedGpus = options?.gpu_options ?? [];
  const resolvedGpus = gpuIds.length ? gpuIds.join(",") : gpuText.trim();

  useEffect(() => {
    if (modelNames.length && !modelNames.includes(model)) setModel(modelNames[0]);
  }, [model, modelNames]);

  const toggleTask = (task: string) => {
    setTasks((current) => current.includes(task) ? current.filter((item) => item !== task) : [...current, task]);
  };

  const toggleGpu = (id: string) => {
    setGpuIds((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]);
    setGpuText("");
  };

  const payload = {
    config,
    models: model ? [model] : [],
    tasks,
    gpus: resolvedGpus,
    parallel: Math.max(1, Number(parallel) || 1),
    max_retries: Math.max(0, Number(maxRetries) || 0),
    no_server: false,
    scoreboard: true,
    rerun,
    dry_run: dryRun,
  };

  const start = async () => {
    if (!connected || !model || !tasks.length || !resolvedGpus) return;
    setStarting(true);
    setStartError(null);
    try {
      onStatus(await api.adminStart(payload));
    } catch (error: unknown) {
      setStartError(error instanceof Error ? error.message : String(error));
    } finally {
      setStarting(false);
    }
  };

  return (
    <section className="task-composer">
      <header>
        <strong>新建评测任务</strong>
        <span>一个任务只部署一个权重，避免不同参数量模型共享 GPU 池。</span>
      </header>

      <div className="task-step">
        <b>01 配置与模型权重</b>
        <div className="task-form-grid">
          <label>
            <span>评测配置</span>
            <select value={config} onChange={(event) => setConfig(event.target.value)}>
              {(configs.length ? configs : ["configs/example.toml"]).map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <label>
            <span>模型</span>
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {modelNames.map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
        </div>
        <dl className="model-deploy-summary">
          <dt>runtime</dt><dd>local vllm-rwkv</dd>
          <dt>weight</dt><dd title={modelDetail?.weight_path ?? ""}>{modelDetail?.weight_path || "从所选 TOML 的 models 配置读取"}</dd>
          <dt>served name</dt><dd>{modelDetail?.served_model_name || model}</dd>
        </dl>
      </div>

      <div className="task-step">
        <b>02 模型权重 → GPU</b>
        {detectedGpus.length ? (
          <div className="gpu-picker">
            {detectedGpus.map((gpu) => (
              <button
                type="button"
                className={gpuIds.includes(gpu.id) ? "selected" : ""}
                key={gpu.id}
                onClick={() => toggleGpu(gpu.id)}
              >
                <strong>GPU {gpu.id}</strong>
                <span>{gpu.name}</span>
                <small>{gpu.memory_used_mib} / {gpu.memory_total_mib} MiB</small>
              </button>
            ))}
          </div>
        ) : (
          <label className="gpu-manual">
            <span>GPU 编号（nvidia-smi 未返回设备时手动指定）</span>
            <input value={gpuText} onChange={(event) => { setGpuText(event.target.value); setGpuIds([]); }} placeholder="例如 0 或 0,1" />
          </label>
        )}
        <div className="binding-line">
          <span>{model || "未选模型"}</span><b>→</b><span>{resolvedGpus ? `GPU ${resolvedGpus}` : "未指定 GPU"}</span>
        </div>
      </div>

      <div className="task-step">
        <b>03 Benchmark（可多选）</b>
        <div className="benchmark-tools">
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索 benchmark" />
          <button type="button" onClick={() => setTasks(visibleTasks)}>选择当前结果</button>
          <button type="button" onClick={() => setTasks([])}>清空</button>
          <span>已选 {tasks.length}</span>
        </div>
        <div className="benchmark-picker">
          {visibleTasks.map((task) => (
            <label className={tasks.includes(task) ? "selected" : ""} key={task}>
              <input type="checkbox" checked={tasks.includes(task)} onChange={() => toggleTask(task)} />
              <span>{task}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="task-step">
        <b>04 运行参数</b>
        <div className="runtime-fields">
          <label><span>并行 slot</span><input type="number" min={1} value={parallel} onChange={(event) => setParallel(event.target.value)} /></label>
          <label><span>失败重试</span><input type="number" min={0} value={maxRetries} onChange={(event) => setMaxRetries(event.target.value)} /></label>
          <label className="task-check"><input type="checkbox" checked={rerun} onChange={(event) => setRerun(event.target.checked)} />覆盖已完成结果</label>
          <label className="task-check"><input type="checkbox" checked={dryRun} onChange={(event) => setDryRun(event.target.checked)} />仅验证计划（dry-run）</label>
        </div>
      </div>

      <div className="task-payload">
        <span>{model || "model"} @ GPU {resolvedGpus || "?"}</span>
        <span>{tasks.length} benchmarks</span>
        <span>{dryRun ? "dry-run" : "真实运行"}</span>
      </div>
      {startError ? <div className="error-bar">{startError}</div> : null}
      <button
        className="task-start"
        type="button"
        disabled={!connected || disabled || starting || !model || !tasks.length || !resolvedGpus}
        onClick={() => void start()}
      >
        {starting ? "正在启动…" : disabled ? "已有任务运行中" : connected ? "启动评测任务" : "等待管理后端连接"}
      </button>
    </section>
  );
}

function RunMonitor({
  status,
  onStatus,
}: {
  status: AdminEvalStatusResponse | null;
  onStatus: (status: AdminEvalStatusResponse) => void;
}) {
  const [actionError, setActionError] = useState<string | null>(null);
  if (!status) {
    return <aside className="run-monitor"><strong>运行状态</strong><div className="muted">等待管理后端…</div></aside>;
  }
  const active = !TERMINAL.has(status.status);
  const pct = Math.round((status.progress_percent || 0) * 100);
  const action = async (request: () => Promise<AdminEvalStatusResponse>) => {
    setActionError(null);
    try {
      onStatus(await request());
    } catch (error: unknown) {
      setActionError(error instanceof Error ? error.message : String(error));
    }
  };
  return (
    <aside className="run-monitor">
      <header><strong>运行状态</strong><span className={`stat-pill ${statusClass(status.status)}`}>{status.status}</span></header>
      <dl>
        <dt>run_id</dt><dd>{status.run_id || "—"}</dd>
        <dt>完成</dt><dd>{status.completed_jobs} / {status.tasks_total}</dd>
        <dt>失败</dt><dd>{status.failed_jobs}</dd>
        <dt>进度</dt><dd>{pct}%</dd>
        <dt>GPU</dt><dd>{status.available_gpus.join(", ") || "—"}</dd>
      </dl>
      <div className="monitor-actions">
        <button type="button" disabled={!active || status.status === "paused"} onClick={() => void action(api.adminPause)}>暂停</button>
        <button type="button" disabled={status.status !== "paused"} onClick={() => void action(api.adminResume)}>恢复</button>
        <button type="button" className="danger" disabled={!active} onClick={() => void action(api.adminCancel)}>取消</button>
      </div>
      {actionError ? <div className="error-bar">{actionError}</div> : null}
      {status.error ? <div className="error-bar">{status.error}</div> : null}
      <div className="queue-summary">
        <b>运行中</b>
        {status.active_jobs.length ? status.active_jobs.map((job) => <span key={job}>{job}</span>) : <span>无</span>}
        <b>等待队列</b>
        {status.queue_head.length ? status.queue_head.map((job) => <span key={job}>{job}</span>) : <span>空</span>}
      </div>
      {status.log_tail.length ? <pre className="admin-log">{status.log_tail.join("\n")}</pre> : null}
    </aside>
  );
}

function TokenField({ onChange }: { onChange: () => void }) {
  const [value, setValue] = useState(getAdminToken());
  return (
    <label className="admin-token">
      <span>Admin Token</span>
      <input
        type="password"
        value={value}
        placeholder="未启用鉴权可留空"
        onChange={(event) => {
          const next = event.target.value.trim();
          setValue(next);
          setAdminToken(next);
        }}
        onBlur={onChange}
      />
    </label>
  );
}
