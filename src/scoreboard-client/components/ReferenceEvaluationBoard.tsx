"use client";

import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import type { EvalContextResponse } from "../lib/dtos/api/eval_context";
import type { EvalRecord } from "../lib/dtos/api/eval_records";
import type {
  LeaderboardMatrix,
  MatrixCell,
  MatrixColumn,
  MatrixDomain,
  MatrixRow,
} from "../lib/dtos/api/leaderboard";

type Generation = "previous" | "current";
type AnswerCategory = "correct" | "incorrect" | "unanswered";

type ParameterGroup = {
  param: string;
  current: MatrixRow;
  previous: MatrixRow | null;
};

type DisplayBenchmark = {
  key: string;
  label: string;
  column: MatrixColumn;
  source: MatrixDomain;
  cellIndex: number;
};

type ScoreSelection = {
  benchmark: DisplayBenchmark;
  group: ParameterGroup;
  generation: Generation;
  model: string;
  cell: MatrixCell;
};

type AnswerRecord = {
  id: string;
  repeatId: number;
  passIndex: number;
  groundTruth: string;
  modelAnswer: string;
  category: AnswerCategory;
  prompt: string;
  completion: string;
  generatedTokens: number;
  latencyMs: number;
  failReason: string;
  source: EvalRecord | null;
};

const EXPERIMENT_TABS = [
  "前代 vs 当代",
  "Prompt template",
  "Fake CoT vs CoT",
  "fp16 vs fp32io16",
  "Qwen3.5 vs RWKV",
] as const;

const DOMAIN_TABS = [
  { key: "overview", label: "常规评估" },
  { key: "math", label: "数学" },
  { key: "knowledge", label: "知识" },
  { key: "instruction_following", label: "指令遵循" },
  { key: "coding", label: "编程" },
  { key: "agent", label: "Agent" },
] as const;

const OVERVIEW_BENCHMARKS = [
  { domain: "math", benchmark: "aime_2024", label: "AIME24" },
  { domain: "math", benchmark: "aime_2025", label: "AIME25" },
  { domain: "math", benchmark: "math_500", label: "MATH-500" },
  { domain: "math", benchmark: "gsm8k_test", label: "GSM8K" },
  { domain: "knowledge", benchmark: "mmlu_test", label: "MMLU" },
  { domain: "instruction_following", benchmark: "ifeval", label: "IFEval" },
] as const;

const CATEGORY_LABELS: Record<AnswerCategory, string> = {
  correct: "正确作答",
  incorrect: "错误作答",
  unanswered: "未能作答",
};

function generationTimestamp(model: string): number {
  const matches = [...model.matchAll(/20\d{6}/g)];
  return Number(matches.at(-1)?.[0] ?? 0);
}

function parameterValue(param: string): number {
  return Number.parseFloat(param.replace(/[^\d.]/g, "")) || 0;
}

function modelGeneration(model: string): string {
  const match = model.match(/-(g\d+[a-z]?)-/i);
  return match?.[1]?.toUpperCase() ?? "RWKV";
}

function groupsFromMatrix(matrix: LeaderboardMatrix): ParameterGroup[] {
  const domain = matrix.domains.find((item) => item.rows.length);
  if (!domain) return [];
  const grouped = new Map<string, MatrixRow[]>();
  domain.rows.forEach((row) => {
    const rows = grouped.get(row.param) ?? [];
    rows.push(row);
    grouped.set(row.param, rows);
  });
  return [...grouped.entries()]
    .map(([param, rows]) => {
      const ordered = rows
        .slice()
        .sort((left, right) => generationTimestamp(right.model) - generationTimestamp(left.model));
      return { param, current: ordered[0], previous: ordered[1] ?? null };
    })
    .sort((left, right) => parameterValue(left.param) - parameterValue(right.param));
}

function displayBenchmarks(matrix: LeaderboardMatrix, domainKey: string): DisplayBenchmark[] {
  if (domainKey === "overview") {
    return OVERVIEW_BENCHMARKS.flatMap((spec) => {
      const source = matrix.domains.find((item) => item.key === spec.domain);
      const cellIndex = source?.columns.findIndex((column) => column.label === spec.benchmark) ?? -1;
      if (!source || cellIndex < 0) return [];
      return [{
        key: `${spec.domain}:${spec.benchmark}`,
        label: spec.label,
        column: source.columns[cellIndex],
        source,
        cellIndex,
      }];
    });
  }
  const source = matrix.domains.find((item) => item.key === domainKey);
  if (!source) return [];
  return source.columns.map((column, cellIndex) => ({
    key: `${source.key}:${column.key}`,
    label: column.label,
    column,
    source,
    cellIndex,
  }));
}

function cellForModel(benchmark: DisplayBenchmark, model: string): MatrixCell | null {
  return benchmark.source.rows.find((row) => row.model === model)?.cells[benchmark.cellIndex] ?? null;
}

function deltaValue(current: MatrixCell | null, previous: MatrixCell | null): number | null {
  if (current?.percent == null || previous?.percent == null) return null;
  return current.percent - previous.percent;
}

function scoreText(value: number | null | undefined): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

function hashString(input: string): number {
  let value = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    value ^= input.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return Math.abs(value);
}

function makeMockAnswers(selection: ScoreSelection): AnswerRecord[] {
  const seed = hashString(`${selection.model}:${selection.benchmark.key}`);
  const isMath = /aime|math|gsm|minerva|olympiad/i.test(selection.benchmark.label);
  const rows: AnswerRecord[] = [];
  (["correct", "incorrect", "unanswered"] as AnswerCategory[]).forEach((category, categoryIndex) => {
    for (let index = 0; index < 10; index += 1) {
      const ordinal = categoryIndex * 10 + index + 1;
      const truth = isMath
        ? String(20 + ((seed + ordinal * 17) % 181))
        : ["A", "B", "C", "D"][(seed + ordinal) % 4];
      const answer = category === "correct"
        ? truth
        : category === "incorrect"
          ? isMath
            ? String(Number(truth) + 1 + (ordinal % 4))
            : ["A", "B", "C", "D"][(seed + ordinal + 1) % 4]
          : "—";
      rows.push({
        id: `${selection.benchmark.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${String(ordinal).padStart(4, "0")}`,
        repeatId: index % 4,
        passIndex: 0,
        groundTruth: truth,
        modelAnswer: answer,
        category,
        prompt: isMath
          ? `请完成 ${selection.benchmark.label} 题目 ${ordinal}，给出最终数值答案。`
          : `请完成 ${selection.benchmark.label} 样本 ${ordinal}，只输出最终选项。`,
        completion: category === "unanswered"
          ? "<think>推理未能在最大生成长度内完成。</think>"
          : `<think>已完成逐步推理并核验结果。</think>\n${answer}`,
        generatedTokens: 168 + ((seed + ordinal * 23) % 240),
        latencyMs: 1800 + ((seed + ordinal * 97) % 2400),
        failReason: category === "correct" ? "" : category === "unanswered" ? "empty completion" : "answer mismatch",
        source: null,
      });
    }
  });
  return rows;
}

function categoryForRecord(record: EvalRecord): AnswerCategory {
  if (record.is_passed) return "correct";
  const diagnostic = `${record.answer || ""} ${record.fail_reason || ""}`.toLowerCase();
  return !record.answer?.trim() || /empty|unanswer|no answer|truncat|max length|未作答|截断/.test(diagnostic)
    ? "unanswered"
    : "incorrect";
}

function answerFromRecord(record: EvalRecord, benchmark: DisplayBenchmark): AnswerRecord {
  return {
    id: `${benchmark.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${String(record.sample_index).padStart(4, "0")}`,
    repeatId: record.repeat_index,
    passIndex: record.pass_index,
    groundTruth: record.ref_answer || "—",
    modelAnswer: record.answer || "—",
    category: categoryForRecord(record),
    prompt: record.context_preview || "完整提示词请打开 detail 查看。",
    completion: record.answer || "—",
    generatedTokens: 0,
    latencyMs: 0,
    failReason: record.fail_reason || "",
    source: record,
  };
}

function truncationRate(selection: ScoreSelection): number {
  return ((hashString(selection.model + selection.benchmark.key) % 12) + 2) / 10;
}

function generationLabel(generation: Generation): string {
  return generation === "current" ? "当代" : "前代";
}

export function ReferenceEvaluationBoard({ matrix }: { matrix: LeaderboardMatrix }) {
  const groups = useMemo(() => groupsFromMatrix(matrix), [matrix]);
  const [experiment, setExperiment] = useState<(typeof EXPERIMENT_TABS)[number]>("前代 vs 当代");
  const [domainKey, setDomainKey] = useState("overview");
  const benchmarks = useMemo(() => displayBenchmarks(matrix, domainKey), [domainKey, matrix]);
  const [selection, setSelection] = useState<ScoreSelection | null>(null);

  useEffect(() => {
    if (selection || !benchmarks.length || !groups.length) return;
    const preferredGroup = groups.find((group) => group.param.toLowerCase() === "2.9b") ?? groups[0];
    const preferredBenchmark = benchmarks.find((item) => /aime25/i.test(item.label)) ?? benchmarks[0];
    const model = preferredGroup.previous?.model ?? preferredGroup.current.model;
    const cell = cellForModel(preferredBenchmark, model);
    if (cell) {
      setSelection({
        benchmark: preferredBenchmark,
        group: preferredGroup,
        generation: preferredGroup.previous ? "previous" : "current",
        model,
        cell,
      });
    }
  }, [benchmarks, groups, selection]);

  const changeDomain = (key: string) => {
    setDomainKey(key);
    setSelection(null);
  };

  return (
    <div className="reference-dashboard">
      <nav className="experiment-tabs" aria-label="实验对比方式">
        {EXPERIMENT_TABS.map((item) => (
          <button
            type="button"
            className={experiment === item ? "active" : ""}
            key={item}
            onClick={() => setExperiment(item)}
          >
            {item}
          </button>
        ))}
      </nav>

      <section className="reference-card comparison-card">
        <nav className="reference-domain-tabs" aria-label="评测领域">
          {DOMAIN_TABS.map((item) => (
            <button
              type="button"
              className={domainKey === item.key ? "active" : ""}
              key={item.key}
              onClick={() => changeDomain(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="comparison-heading">
          <div>
            <strong>{domainKey === "overview" ? "常规评估" : DOMAIN_TABS.find((item) => item.key === domainKey)?.label} · {experiment}</strong>
            <span><b>前代 → 当代</b> · 仅改变模型代际；prompt、precision、sampling 与输出边界保持一致。</span>
          </div>
          <span className="temporary-data-badge">临时展示数据</span>
        </div>

        <div className="reference-table-scroll">
          <table className="reference-score-table">
            <thead>
              <tr>
                <th className="ref-benchmark" rowSpan={2}>benchmark</th>
                <th className="ref-samples" rowSpan={2}>n_<br />samples</th>
                <th className="ref-metric" rowSpan={2}>k_<br />metric</th>
                {groups.map((group) => <th colSpan={3} key={group.param}>{group.param.toUpperCase()}</th>)}
              </tr>
              <tr>
                {groups.flatMap((group) => [
                  <th key={`${group.param}:previous`}>前代</th>,
                  <th key={`${group.param}:current`}>当代</th>,
                  <th className="ref-delta-head" key={`${group.param}:delta`}>delta</th>,
                ])}
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((benchmark, benchmarkIndex) => (
                <tr className={`reference-row-tone tone-${benchmarkIndex % 4}`} key={benchmark.key}>
                  <td className="ref-benchmark">{benchmark.label}</td>
                  <td className="ref-samples">{benchmark.column.num_samples?.toLocaleString() ?? "—"}</td>
                  <td className="ref-metric">{benchmark.column.metric ?? "score"}</td>
                  {groups.flatMap((group) => {
                    const previous = group.previous ? cellForModel(benchmark, group.previous.model) : null;
                    const current = cellForModel(benchmark, group.current.model);
                    const delta = deltaValue(current, previous);
                    return [
                      <td key={`${benchmark.key}:${group.param}:previous`}>
                        <CompactScore
                          cell={previous}
                          selected={selection?.benchmark.key === benchmark.key
                            && selection.group.param === group.param
                            && selection.generation === "previous"}
                          onClick={previous && group.previous ? () => setSelection({
                            benchmark,
                            group,
                            generation: "previous",
                            model: group.previous!.model,
                            cell: previous,
                          }) : undefined}
                        />
                      </td>,
                      <td key={`${benchmark.key}:${group.param}:current`}>
                        <CompactScore
                          cell={current}
                          selected={selection?.benchmark.key === benchmark.key
                            && selection.group.param === group.param
                            && selection.generation === "current"}
                          onClick={current ? () => setSelection({
                            benchmark,
                            group,
                            generation: "current",
                            model: group.current.model,
                            cell: current,
                          }) : undefined}
                        />
                      </td>,
                      <td className={`ref-delta ${delta != null && delta > 0.05 ? "positive" : delta != null && delta < -0.05 ? "negative" : ""}`} key={`${benchmark.key}:${group.param}:delta`}>
                        {delta == null ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%`}
                      </td>,
                    ];
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {selection ? (
        <AnswerDetails selection={selection} onClear={() => setSelection(null)} />
      ) : (
        <section className="reference-card selection-empty">
          点击表格中的前代或当代分数，查看该模型的逐题作答详情。
        </section>
      )}
    </div>
  );
}

function CompactScore({
  cell,
  selected,
  onClick,
}: {
  cell: MatrixCell | null;
  selected: boolean;
  onClick?: () => void;
}) {
  if (cell?.percent == null) return <span className="compact-score-missing">—</span>;
  const standard = Math.max(0, Math.min(100, cell.percent));
  const potential = cell.potential_percent == null
    ? null
    : Math.max(standard, Math.min(100, cell.potential_percent));
  return (
    <button
      type="button"
      className={`compact-score${selected ? " selected" : ""}${potential == null ? "" : " has-potential"}`}
      disabled={!onClick}
      onClick={onClick}
      title={potential == null
        ? `标准分 ${scoreText(standard)}`
        : `标准分 ${scoreText(standard)}；潜力分 ${scoreText(potential)}`}
    >
      <span>{standard.toFixed(1)}{potential == null ? "" : <small> ({potential.toFixed(1)})</small>}%</span>
    </button>
  );
}

function AnswerDetails({ selection, onClear }: { selection: ScoreSelection; onClear: () => void }) {
  const [category, setCategory] = useState<AnswerCategory>("correct");
  const [contextAnswer, setContextAnswer] = useState<AnswerRecord | null>(null);
  const [databaseRecords, setDatabaseRecords] = useState<EvalRecord[] | null>(null);
  const [recordsLoading, setRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const taskId = selection.cell.meta?.task_id ?? null;
  const answers = useMemo(
    () => databaseRecords === null
      ? makeMockAnswers(selection)
      : databaseRecords.map((record) => answerFromRecord(record, selection.benchmark)),
    [databaseRecords, selection],
  );
  const visibleAnswers = answers.filter((answer) => answer.category === category);
  const estimatedTotal = Math.max(
    10,
    Math.round((selection.cell.num_samples ?? selection.benchmark.column.num_samples ?? 100)
      * (category === "correct"
        ? (selection.cell.percent ?? 0) / 100
        : category === "incorrect"
          ? Math.max(0, 1 - (selection.cell.percent ?? 0) / 100) * 0.82
          : 0.04)),
  );
  const accuracy = selection.cell.percent ?? 0;
  const truncation = truncationRate(selection);
  const metric = selection.cell.metric ?? selection.benchmark.column.metric ?? "score";
  const sampleCount = selection.cell.num_samples ?? selection.benchmark.column.num_samples ?? 0;

  useEffect(() => {
    setCategory("correct");
    setContextAnswer(null);
  }, [selection]);

  useEffect(() => {
    if (taskId === null) {
      setDatabaseRecords(null);
      setRecordsError(null);
      setRecordsLoading(false);
      return;
    }
    let cancelled = false;
    setDatabaseRecords([]);
    setRecordsLoading(true);
    setRecordsError(null);
    api.evalRecords(taskId, false, 200, 0)
      .then((payload) => {
        if (!cancelled) setDatabaseRecords(payload.records);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setRecordsError(error instanceof Error ? error.message : String(error));
          setDatabaseRecords([]);
        }
      })
      .finally(() => {
        if (!cancelled) setRecordsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  return (
    <>
      <section className="reference-card answer-details" id="answer-details">
        <header className="answer-details-head">
          <strong>作答详情</strong>
          <button type="button" onClick={onClear}>清除选择</button>
        </header>

        <div className="selection-chips">
          <span>RWKV</span>
          <span title={selection.model}>{modelGeneration(selection.model)}</span>
          <span>{selection.group.param.toUpperCase()}</span>
          <span>{selection.benchmark.label}</span>
          <span>n={sampleCount}</span>
          <span>{metric}</span>
          <span className="warning">截断率：{truncation.toFixed(1)}%</span>
          <span className="success">准确率：{accuracy.toFixed(1)}%</span>
          {selection.cell.potential_percent == null ? null : (
            <span className="potential-chip">潜力：{selection.cell.potential_percent.toFixed(1)}%</span>
          )}
          <span className={taskId === null ? "warning" : "success"}>
            {taskId === null ? "模拟明细" : `数据库 task_id=${taskId}`}
          </span>
        </div>

        <div className="configuration-row">
          <div className="configuration-card prompt-card">
            <small>prompt_template</small>
            <pre>User❉&#123;task.problem&#125;❉{"\n"}Bot❉&lt;think</pre>
          </div>
          <div className="configuration-card sampling-card">
            <small>sampling_config</small>
            <div>
              <ConfigValue label="temperature" value="0.6" />
              <ConfigValue label="top_p" value="0.95" />
              <ConfigValue label="top_k" value="40" />
              <ConfigValue label="max_tokens" value="32768" />
              <ConfigValue label="seed" value="42" />
            </div>
          </div>
        </div>

        <nav className="answer-category-tabs" aria-label="作答结果分类">
          {(Object.keys(CATEGORY_LABELS) as AnswerCategory[]).map((item) => (
            <button
              type="button"
              className={category === item ? "active" : ""}
              key={item}
              onClick={() => setCategory(item)}
            >
              {CATEGORY_LABELS[item]} <span>{answers.filter((answer) => answer.category === item).length}</span>
            </button>
          ))}
        </nav>

        {recordsLoading ? <div className="spinner">正在读取数据库作答明细…</div> : null}
        {recordsError ? <div className="error-bar">数据库明细加载失败：{recordsError}</div> : null}

        {taskId === null ? <p className="answer-sampling-note">
          从该结果类别的 {estimatedTotal} 条记录中随机抽取 <strong>{visibleAnswers.length}</strong> 条
        </p> : <p className="answer-sampling-note">
          当前已读取 <strong>{answers.length}</strong> 条数据库记录，本类显示 <strong>{visibleAnswers.length}</strong> 条
        </p>}

        <div className="answer-table-wrap">
          <table className="answer-table">
            <thead>
              <tr>
                <th>题目 ID</th>
                <th>repeat_id</th>
                <th>ground_truth</th>
                <th>模型作答（判分器提取）</th>
                <th>is_passed</th>
                <th>detail</th>
              </tr>
            </thead>
            <tbody>
              {visibleAnswers.map((answer, index) => (
                <tr className={`answer-tone-${index % 4}`} key={answer.id}>
                  <td>{answer.id}</td>
                  <td>{answer.repeatId}</td>
                  <td>{answer.groundTruth}</td>
                  <td>{answer.modelAnswer}</td>
                  <td>
                    <span className={`answer-outcome ${answer.category === "correct" ? "pass" : answer.category === "incorrect" ? "fail" : "empty"}`}>
                      {answer.category === "correct" ? "true" : answer.category === "incorrect" ? "false" : "null"}
                    </span>
                  </td>
                  <td><button className="answer-detail-button" type="button" onClick={() => setContextAnswer(answer)}>detail</button></td>
                </tr>
              ))}
              {!recordsLoading && visibleAnswers.length === 0 ? (
                <tr><td colSpan={6} className="muted">该分类暂无记录。</td></tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      {contextAnswer ? (
        <FullContextModal
          answer={contextAnswer}
          selection={selection}
          accuracy={accuracy}
          truncation={truncation}
          taskId={taskId}
          onClose={() => setContextAnswer(null)}
        />
      ) : null}
    </>
  );
}

function ConfigValue({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  );
}

function FullContextModal({
  answer,
  selection,
  accuracy,
  truncation,
  taskId,
  onClose,
}: {
  answer: AnswerRecord;
  selection: ScoreSelection;
  accuracy: number;
  truncation: number;
  taskId: number | null;
  onClose: () => void;
}) {
  const [databaseContext, setDatabaseContext] = useState<EvalContextResponse | null>(null);
  const [contextError, setContextError] = useState<string | null>(null);

  useEffect(() => {
    if (taskId === null || answer.source === null) {
      setDatabaseContext(null);
      setContextError(null);
      return;
    }
    let cancelled = false;
    api.evalContext(taskId, answer.source.sample_index, answer.source.repeat_index, answer.source.pass_index)
      .then((payload) => {
        if (!cancelled) setDatabaseContext(payload);
      })
      .catch((error: unknown) => {
        if (!cancelled) setContextError(error instanceof Error ? error.message : String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [answer.source, taskId]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  const passed = answer.category === "correct";
  const contextText = databaseContext?.view === "text"
    ? databaseContext.raw_text
    : databaseContext?.context
      ? JSON.stringify(databaseContext.context, null, 2)
      : null;
  return (
    <div className="reference-modal-backdrop" onClick={onClose}>
      <div className="reference-modal" onClick={(event) => event.stopPropagation()}>
        <header>
          <strong>完整模型上下文</strong>
          <button type="button" onClick={onClose}>关闭</button>
        </header>
        <div className="reference-modal-grid">
          <div className="context-main">
            {contextError ? <div className="error-bar">数据库 context 加载失败：{contextError}</div> : null}
            {taskId !== null && answer.source !== null && !databaseContext && !contextError
              ? <div className="spinner">正在读取完整模型上下文…</div>
              : null}
            {contextText ? <ContextBlock label="database context" value={contextText} /> : null}
            <ContextBlock label="assembled prompt" value={`User❉${answer.prompt}❉\nBot❉<think`} />
            <ContextBlock label="raw completion" value={answer.completion} />
          </div>
          <aside className="context-sidebar">
            <h3>基础信息</h3>
            <div className="selection-chips compact">
              <span>RWKV</span>
              <span>{modelGeneration(selection.model)}</span>
              <span>{selection.group.param.toUpperCase()}</span>
              <span>{selection.benchmark.label}</span>
              <span>n={selection.cell.num_samples ?? selection.benchmark.column.num_samples ?? 0}</span>
              <span>{selection.cell.metric ?? "score"}</span>
              <span className="warning">截断率：{truncation.toFixed(1)}%</span>
              <span className="success">准确率：{accuracy.toFixed(1)}%</span>
            </div>
            <dl className="context-key-values">
              <dt>problem_id</dt><dd>{answer.id}</dd>
              <dt>repeat_id</dt><dd>{answer.repeatId}</dd>
              <dt>pass_index</dt><dd>{answer.passIndex}</dd>
              <dt>task_id</dt><dd>{taskId ?? "mock"}</dd>
              <dt>generation</dt><dd>{generationLabel(selection.generation)}</dd>
              <dt>is_passed</dt><dd><span className={`answer-outcome ${passed ? "pass" : "fail"}`}>{passed ? "true" : "false"}</span></dd>
            </dl>
            <ContextBlock label="prompt_template" value={"User❉{task.problem}❉\nBot❉<think"} />
            <div className="modal-sampling">
              <small>sampling_config</small>
              <div>
                <ConfigValue label="temperature" value="0.6" />
                <ConfigValue label="top_p" value="0.95" />
                <ConfigValue label="top_k" value="40" />
                <ConfigValue label="max_tokens" value="32768" />
                <ConfigValue label="seed" value="42" />
              </div>
            </div>
            <h3>SCORING RESULT</h3>
            <dl className="context-key-values">
              <dt>ground_truth</dt><dd>{answer.groundTruth}</dd>
              <dt>extracted_answer</dt><dd>{answer.modelAnswer}</dd>
              <dt>fail_reason</dt><dd>{passed ? "—" : answer.category === "unanswered" ? "empty completion" : "answer mismatch"}</dd>
            </dl>
            <h3>GENERATION METADATA</h3>
            <dl className="context-key-values">
              <dt>run_id</dt><dd>mock-generation-{selection.group.param}-{answer.repeatId}</dd>
              <dt>generated_tokens</dt><dd>{answer.generatedTokens}</dd>
              <dt>latency_ms</dt><dd>{answer.latencyMs}</dd>
            </dl>
          </aside>
        </div>
      </div>
    </div>
  );
}

function ContextBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="context-block">
      <small>{label}</small>
      <pre>{value}</pre>
    </div>
  );
}
