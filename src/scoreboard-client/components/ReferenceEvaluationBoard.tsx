"use client";

import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import {
  CORE_BENCHMARK_CATALOG,
  benchmarkMatches,
} from "../lib/benchmarkCatalog";
import type { BenchmarkCatalogItem } from "../lib/benchmarkCatalog";
import type { EvalContextResponse } from "../lib/dtos/api/eval_context";
import type { EvalRecord, EvalRecordsResponse } from "../lib/dtos/api/eval_records";
import type {
  LeaderboardMatrix,
  MatrixCell,
  MatrixColumn,
  MatrixDomain,
  MatrixRow,
} from "../lib/dtos/api/leaderboard";

type Architecture = `g1${string}`;
type EvalMode = "CoT" | "NoCoT";
type AnswerCategory = "all" | "correct" | "incorrect" | "unanswered";
type ClassifiedAnswerCategory = Exclude<AnswerCategory, "all">;

type ParameterGroup = {
  param: string;
  models: Array<{
    architecture: Architecture;
    row: MatrixRow;
  }>;
};

type DisplayBenchmark = {
  key: string;
  label: string;
  column: MatrixColumn;
  source: MatrixDomain | null;
  cellIndex: number;
  evalMode: EvalMode;
  deferred?: string;
};

type ScoreSelection = {
  benchmark: DisplayBenchmark;
  group: ParameterGroup;
  architecture: Architecture;
  model: string;
  cell: MatrixCell;
};

type AnswerRecord = {
  key: string;
  id: string;
  sampleIndex: number;
  repeatId: number;
  passIndex: number;
  groundTruth: string;
  modelAnswer: string;
  category: ClassifiedAnswerCategory;
  prompt: string;
  completion: string;
  failReason: string;
  source: EvalRecord;
};

const EXPERIMENT_TABS = [
  "前代 vs 当代",
  "Prompt template",
  "Fake CoT vs CoT",
  "fp16 vs fp32io16",
  "Qwen3.5 vs RWKV",
] as const;

const DOMAIN_TABS = [
  { key: "all", label: "常规评估" },
  { key: "math", label: "数学" },
  { key: "knowledge", label: "知识" },
  { key: "instruction_following", label: "指令遵循" },
  { key: "coding", label: "编程" },
  { key: "function_call", label: "FC" },
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
  all: "全部作答",
  correct: "正确作答",
  incorrect: "错误作答",
  unanswered: "未能作答",
};

const ANSWER_PAGE_SIZE = 20;

function generationTimestamp(model: string): number {
  const matches = [...model.matchAll(/20\d{6}/g)];
  return Number(matches.at(-1)?.[0] ?? 0);
}

function parameterValue(param: string): number {
  return Number.parseFloat(param.replace(/[^\d.]/g, "")) || 0;
}

function modelArchitecture(model: string): Architecture | null {
  const match = model.match(/-(g1[a-z])(?:-|_)/i);
  const architecture = match?.[1]?.toLowerCase();
  return architecture ? architecture as Architecture : null;
}

function modelGeneration(model: string): string {
  const architecture = modelArchitecture(model);
  return architecture ? architectureLabel(architecture) : "RWKV";
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
      const architectures = new Map<Architecture, MatrixRow>();
      ordered.forEach((row) => {
        const architecture = modelArchitecture(row.model);
        if (architecture && !architectures.has(architecture)) {
          architectures.set(architecture, row);
        }
      });
      const models = [...architectures.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([architecture, row]) => ({ architecture, row }))
        .slice(-2);
      return { param, models };
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
        evalMode: normalizedEvalMode(source.columns[cellIndex].eval_method) === "cot" ? "CoT" : "NoCoT",
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
    evalMode: normalizedEvalMode(column.eval_method) === "cot" ? "CoT" : "NoCoT",
  }));
}

function catalogBenchmarks(matrix: LeaderboardMatrix, domainKey: string): DisplayBenchmark[] {
  const requested = domainKey === "all"
    ? CORE_BENCHMARK_CATALOG
    : CORE_BENCHMARK_CATALOG.filter((item) => item.domain === domainKey);
  return requested.flatMap((spec) => {
    const modes: EvalMode[] = spec.domain === "instruction_following"
      ? ["NoCoT"]
      : ["NoCoT", "CoT"];
    return modes.map((evalMode) => displayCatalogBenchmark(matrix, spec, evalMode));
  });
}

function benchmarksForDomain(matrix: LeaderboardMatrix, domainKey: string): DisplayBenchmark[] {
  return domainKey === "function_call"
    ? displayBenchmarks(matrix, domainKey)
    : catalogBenchmarks(matrix, domainKey);
}

function normalizedEvalMode(value: string): string {
  return value.toLowerCase().replace(/[^a-z]/g, "");
}

function displayCatalogBenchmark(
  matrix: LeaderboardMatrix,
  spec: BenchmarkCatalogItem,
  evalMode: EvalMode,
): DisplayBenchmark {
  const preferred = matrix.domains.find((item) => item.key === spec.domain) ?? null;
  const sources = preferred
    ? [preferred, ...matrix.domains.filter((item) => item !== preferred)]
    : matrix.domains;
  for (const source of sources) {
    const cellIndex = source.columns.findIndex(
      (column) => (
        (benchmarkMatches(spec, column.key.split(":")[0]) || benchmarkMatches(spec, column.label))
        && normalizedEvalMode(column.eval_method) === normalizedEvalMode(evalMode)
      ),
    );
    if (cellIndex >= 0) {
      return {
        key: `${spec.key}:${evalMode.toLowerCase()}`,
        label: spec.label,
        column: source.columns[cellIndex],
        source,
        cellIndex,
        evalMode,
        deferred: spec.deferred,
      };
    }
  }
  return {
    key: `${spec.key}:${evalMode.toLowerCase()}`,
    label: spec.label,
    column: {
      key: `${spec.key}:${evalMode.toLowerCase()}`,
      label: spec.label,
      metric: null,
      eval_method: evalMode,
      num_samples: spec.samples,
    },
    source: null,
    cellIndex: -1,
    evalMode,
    deferred: spec.deferred,
  };
}

function cellForModel(benchmark: DisplayBenchmark, model: string): MatrixCell | null {
  if (!benchmark.source || benchmark.cellIndex < 0) return null;
  const cell = benchmark.source.rows.find((row) => row.model === model)?.cells[benchmark.cellIndex] ?? null;
  const cellMode = cell?.meta?.eval_method;
  if (cell?.percent == null || !cellMode) return null;
  return normalizedEvalMode(cellMode) === normalizedEvalMode(benchmark.evalMode) ? cell : null;
}

function scoreText(value: number | null | undefined): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

function rateText(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function deltaValue(current: MatrixCell | null, previous: MatrixCell | null): number | null {
  if (current?.percent == null || previous?.percent == null) return null;
  return current.percent - previous.percent;
}

function categoryForRecord(record: EvalRecord): ClassifiedAnswerCategory {
  if (record.is_passed) return "correct";
  const diagnostic = `${record.answer || ""} ${record.fail_reason || ""}`.toLowerCase();
  return !record.answer?.trim() || /empty|unanswer|no answer|truncat|max length|未作答|截断/.test(diagnostic)
    ? "unanswered"
    : "incorrect";
}

function answerFromRecord(record: EvalRecord, benchmark: DisplayBenchmark): AnswerRecord {
  return {
    key: `${benchmark.key}:${record.sample_index}:${record.repeat_index}:${record.pass_index}`,
    id: `${benchmark.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${String(record.sample_index).padStart(4, "0")}`,
    sampleIndex: record.sample_index,
    repeatId: record.repeat_index,
    passIndex: record.pass_index,
    groundTruth: record.ref_answer || "—",
    modelAnswer: record.answer || "—",
    category: categoryForRecord(record),
    prompt: record.context_preview || "完整提示词请打开 detail 查看。",
    completion: record.answer || "—",
    failReason: record.fail_reason || "",
    source: record,
  };
}

function architectureLabel(architecture: Architecture): string {
  return `G1${architecture.at(-1)}`;
}

export function ReferenceEvaluationBoard({
  matrix,
  initialDomain = "all",
}: {
  matrix: LeaderboardMatrix;
  initialDomain?: string;
}) {
  const groups = useMemo(() => groupsFromMatrix(matrix), [matrix]);
  const [experiment, setExperiment] = useState<(typeof EXPERIMENT_TABS)[number]>("前代 vs 当代");
  const [domainKey, setDomainKey] = useState(
    DOMAIN_TABS.some((item) => item.key === initialDomain) ? initialDomain : "all",
  );
  const benchmarks = useMemo(() => benchmarksForDomain(matrix, domainKey), [domainKey, matrix]);
  const dataColumnCount = groups.reduce(
    (total, group) => total + group.models.length + 1,
    0,
  );
  const dataColumnWidth = dataColumnCount > 0 ? `${85.06 / dataColumnCount}%` : "auto";
  const domainTabs = useMemo(
    () => DOMAIN_TABS.map((item) => ({
      ...item,
      count: item.key === "all"
        ? CORE_BENCHMARK_CATALOG.length
        : item.key === "function_call"
          ? displayBenchmarks(matrix, item.key).length
          : CORE_BENCHMARK_CATALOG.filter((benchmark) => benchmark.domain === item.key).length,
    })),
    [matrix],
  );
  const [selection, setSelection] = useState<ScoreSelection | null>(null);

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
          {domainTabs.map((item) => (
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

        <div className="reference-table-scroll">
          <table className="reference-score-table">
            <colgroup>
              <col className="ref-col-benchmark" />
              <col className="ref-col-samples" />
              <col className="ref-col-metric" />
              {groups.flatMap((group) => [
                ...group.models.map(({ architecture }) => (
                  <col
                    className="ref-col-score"
                    key={`${group.param}:${architecture}`}
                    style={{ width: dataColumnWidth }}
                  />
                )),
                <col
                  className="ref-col-delta"
                  key={`${group.param}:delta`}
                  style={{ width: dataColumnWidth }}
                />,
              ])}
            </colgroup>
            <thead>
              <tr>
                <th className="ref-benchmark" rowSpan={2}>benchmark</th>
                <th className="ref-samples" rowSpan={2}>n_<br />samples</th>
                <th className="ref-metric" rowSpan={2}>k_<br />metric</th>
                {groups.map((group) => <th colSpan={group.models.length + 1} key={group.param}>{group.param.toUpperCase()}</th>)}
              </tr>
              <tr>
                {groups.flatMap((group) => [
                  ...group.models.map(({ architecture }) => (
                    <th key={`${group.param}:${architecture}`}>{architectureLabel(architecture)}</th>
                  )),
                  <th className="ref-delta-head" key={`${group.param}:delta`}>delta</th>,
                ])}
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((benchmark, benchmarkIndex) => (
                <tr className={`reference-row-tone tone-${benchmarkIndex % 4}`} key={benchmark.key}>
                  <td className="ref-benchmark">
                    {benchmark.label}
                    <small className={`benchmark-mode ${benchmark.evalMode.toLowerCase()}`}>{benchmark.evalMode}</small>
                    {benchmark.deferred ? <small className="benchmark-deferred">（{benchmark.deferred}）</small> : null}
                  </td>
                  <td className="ref-samples">{benchmark.column.num_samples?.toLocaleString() ?? "—"}</td>
                  <td className="ref-metric">{benchmark.column.metric ?? "—"}</td>
                  {groups.flatMap((group) => {
                    const scoreCells = group.models.map(({ row }) => cellForModel(benchmark, row.model));
                    const delta = deltaValue(scoreCells.at(-1) ?? null, scoreCells.at(-2) ?? null);
                    return [
                      ...group.models.map(({ architecture, row }, modelIndex) => {
                        const cell = scoreCells[modelIndex];
                        return (
                          <td key={`${benchmark.key}:${group.param}:${architecture}`}>
                            <CompactScore
                              cell={cell}
                              selected={selection?.benchmark.key === benchmark.key
                                && selection.group.param === group.param
                                && selection.architecture === architecture}
                              onClick={cell?.meta?.task_id != null ? () => setSelection({
                                benchmark,
                                group,
                                architecture,
                                model: row.model,
                                cell,
                              }) : undefined}
                            />
                          </td>
                        );
                      }),
                      <td
                        className={`ref-delta ${delta != null && delta > 0.05 ? "positive" : delta != null && delta < -0.05 ? "negative" : ""}`}
                        key={`${benchmark.key}:${group.param}:delta`}
                      >
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
          点击表格中的实测分数，查看该架构模型的逐题作答详情。
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
      className={`compact-score${selected ? " selected" : ""}${potential == null ? "" : " has-potential"}${onClick ? "" : " no-detail"}`}
      disabled={!onClick}
      onClick={onClick}
      title={!onClick
        ? `标准分 ${scoreText(standard)}；暂无逐题明细`
        : potential == null
          ? `标准分 ${scoreText(standard)}；点击查看逐题明细`
          : `标准分 ${scoreText(standard)}；潜力分 ${scoreText(potential)}；点击查看逐题明细`}
    >
      <span>{standard.toFixed(1)}{potential == null ? "" : <small> ({potential.toFixed(1)})</small>}%</span>
    </button>
  );
}

function AnswerDetails({ selection, onClear }: { selection: ScoreSelection; onClear: () => void }) {
  const [category, setCategory] = useState<AnswerCategory>("all");
  const [contextAnswer, setContextAnswer] = useState<AnswerRecord | null>(null);
  const [databaseRecords, setDatabaseRecords] = useState<EvalRecord[] | null>(null);
  const [recordsSummary, setRecordsSummary] = useState<EvalRecordsResponse | null>(null);
  const [recordsPage, setRecordsPage] = useState(0);
  const [recordsHasMore, setRecordsHasMore] = useState(false);
  const [recordsLoading, setRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const taskId = selection.cell.meta?.task_id ?? null;
  const answers = useMemo(
    () => (databaseRecords ?? []).map((record) => answerFromRecord(record, selection.benchmark)),
    [databaseRecords, selection],
  );
  const visibleAnswers = answers;
  const accuracy = selection.cell.percent ?? 0;
  const metric = selection.cell.metric ?? selection.benchmark.column.metric ?? "score";
  const sampleCount = selection.cell.num_samples ?? selection.benchmark.column.num_samples ?? 0;

  useEffect(() => {
    setCategory("all");
    setContextAnswer(null);
    setRecordsSummary(null);
    setRecordsPage(0);
  }, [selection]);

  useEffect(() => {
    if (taskId === null) {
      setDatabaseRecords([]);
      setRecordsSummary(null);
      setRecordsHasMore(false);
      setRecordsError("该分数没有关联数据库 task_id，无法读取逐题明细。");
      setRecordsLoading(false);
      return;
    }
    let cancelled = false;
    setDatabaseRecords([]);
    setRecordsSummary(null);
    setRecordsLoading(true);
    setRecordsError(null);
    api.evalRecords(taskId, false, ANSWER_PAGE_SIZE, recordsPage * ANSWER_PAGE_SIZE, category)
      .then((payload) => {
        if (!cancelled) {
          setDatabaseRecords(payload.records);
          setRecordsSummary(payload);
          setRecordsHasMore(payload.has_more);
        }
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
  }, [category, recordsPage, taskId]);

  const recordStart = recordsSummary && recordsSummary.records.length
    ? recordsSummary.offset + 1
    : 0;
  const recordEnd = recordsSummary?.next_offset ?? 0;

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
          <span className="success">准确率：{accuracy.toFixed(1)}%</span>
          {selection.cell.potential_percent == null ? null : (
            <span className="potential-chip">潜力：{selection.cell.potential_percent.toFixed(1)}%</span>
          )}
          <span className="success">task_id={taskId}</span>
        </div>

        {recordsSummary ? (
          <div className="answer-final-diagnostics" aria-label="任务级最终回答诊断">
            <div className="answer-final-diagnostic primary">
              <small>最终回答截断率</small>
              <strong>{rateText(recordsSummary.diagnostics.truncation_rate)}</strong>
              <span>
                {recordsSummary.diagnostics.truncated_count} / {recordsSummary.completion_total} completions
              </span>
            </div>
            <div className="answer-final-diagnostic">
              <small>最终停止原因覆盖</small>
              <strong>
                {recordsSummary.completion_total
                  ? rateText(recordsSummary.diagnostics.final_stop_telemetry_count / recordsSummary.completion_total)
                  : "—"}
              </strong>
              <span>
                {recordsSummary.diagnostics.final_stop_telemetry_count} / {recordsSummary.completion_total} completions
              </span>
            </div>
            <div className="answer-final-diagnostic">
              <small>Completion / Eval</small>
              <strong>{recordsSummary.completion_total} / {recordsSummary.eval_total}</strong>
              <span>missing eval: {recordsSummary.missing_eval_count}</span>
            </div>
            <p className="answer-final-diagnostic-scope">
              仅统计提交给评测器的最终回答阶段；Math 只统计第二阶段答案，不计第一阶段推理截断。
            </p>
          </div>
        ) : null}

        <nav className="answer-category-tabs" aria-label="作答结果分类">
          {(Object.keys(CATEGORY_LABELS) as AnswerCategory[]).map((item) => (
            <button
              type="button"
              className={category === item ? "active" : ""}
              key={item}
              onClick={() => {
                setCategory(item);
                setRecordsPage(0);
              }}
            >
              {CATEGORY_LABELS[item]} <span>{recordsSummary?.outcome_counts[item] ?? 0}</span>
            </button>
          ))}
        </nav>

        {recordsLoading ? <div className="spinner">正在读取作答明细…</div> : null}
        {recordsError ? <div className="error-bar">作答明细加载失败：{recordsError}</div> : null}

        <p className="answer-sampling-note">
          第 <strong>{recordStart}–{recordEnd}</strong> / <strong>{recordsSummary?.filtered_total ?? 0}</strong> 条，
          当前页显示 <strong>{visibleAnswers.length}</strong> 条
        </p>

        <div className="answer-table-wrap">
          <table className="answer-table">
            <colgroup>
              <col className="answer-col-id" />
              <col className="answer-col-repeat" />
              <col className="answer-col-pass" />
              <col className="answer-col-ground-truth" />
              <col className="answer-col-model-answer" />
              <col className="answer-col-outcome" />
              <col className="answer-col-stop" />
              <col className="answer-col-detail" />
            </colgroup>
            <thead>
              <tr>
                <th>题目 ID</th>
                <th>repeat_id</th>
                <th>pass_index</th>
                <th>ground_truth</th>
                <th>模型作答（判分器提取）</th>
                <th>is_passed</th>
                <th>final_stop</th>
                <th>detail</th>
              </tr>
            </thead>
            <tbody>
              {visibleAnswers.map((answer, index) => (
                <tr className={`answer-tone-${index % 4}`} key={answer.key}>
                  <td>{answer.id}</td>
                  <td>{answer.repeatId}</td>
                  <td>{answer.passIndex}</td>
                  <td>{answer.groundTruth}</td>
                  <td>{answer.modelAnswer}</td>
                  <td>
                    <span className={`answer-outcome ${answer.category === "correct" ? "pass" : answer.category === "incorrect" ? "fail" : "empty"}`}>
                      {answer.category === "correct" ? "true" : answer.category === "incorrect" ? "false" : "null"}
                    </span>
                  </td>
                  <td>
                    <span className={`answer-final-stop ${answer.source.is_truncated ? "truncated" : ""}`}>
                      {answer.source.final_stop_reason || "—"}
                    </span>
                  </td>
                  <td><button className="answer-detail-button" type="button" onClick={() => setContextAnswer(answer)}>detail</button></td>
                </tr>
              ))}
              {!recordsLoading && visibleAnswers.length === 0 ? (
                <tr><td colSpan={8} className="muted">该分类暂无记录。</td></tr>
              ) : null}
            </tbody>
          </table>
        </div>
        {taskId !== null ? (
          <nav className="answer-pagination" aria-label="作答详情分页">
            <button
              type="button"
              disabled={recordsPage === 0 || recordsLoading}
              onClick={() => setRecordsPage((page) => Math.max(0, page - 1))}
            >
              上一页
            </button>
            <span>第 {recordsPage + 1} 页 · 每页 {ANSWER_PAGE_SIZE} 题</span>
            <button
              type="button"
              disabled={!recordsHasMore || recordsLoading}
              onClick={() => setRecordsPage((page) => page + 1)}
            >
              下一页
            </button>
          </nav>
        ) : null}
      </section>

      {contextAnswer && taskId !== null ? (
        <FullContextModal
          answer={contextAnswer}
          selection={selection}
          accuracy={accuracy}
          taskId={taskId}
          onClose={() => setContextAnswer(null)}
        />
      ) : null}
    </>
  );
}

type ContextStage = {
  prompt: string;
  completion: string;
  stopReason: string;
};

function objectValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function readableValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === undefined || value === null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function stagesFromContext(payload: EvalContextResponse | null): ContextStage[] {
  const stages = payload?.context?.stages;
  if (!Array.isArray(stages)) return [];
  return stages.flatMap((value) => {
    const stage = objectValue(value);
    if (!stage) return [];
    return [{
      prompt: readableValue(stage.prompt),
      completion: readableValue(stage.completion),
      stopReason: readableValue(stage.stop_reason),
    }];
  });
}

function remainingStructuredContext(payload: EvalContextResponse | null): Record<string, unknown> | null {
  if (!payload?.context) return null;
  const entries = Object.entries(payload.context)
    .filter(([key]) => key !== "stages" && key !== "sampling_config");
  return entries.length ? Object.fromEntries(entries) : null;
}

function FullContextModal({
  answer,
  selection,
  accuracy,
  taskId,
  onClose,
}: {
  answer: AnswerRecord;
  selection: ScoreSelection;
  accuracy: number;
  taskId: number;
  onClose: () => void;
}) {
  const [databaseContext, setDatabaseContext] = useState<EvalContextResponse | null>(null);
  const [contextLoading, setContextLoading] = useState(false);
  const [contextError, setContextError] = useState<string | null>(null);

  useEffect(() => {
    setDatabaseContext(null);
    setContextError(null);
    let cancelled = false;
    setContextLoading(true);
    api.evalContext(taskId, answer.source.sample_index, answer.source.repeat_index, answer.source.pass_index)
      .then((payload) => {
        if (!cancelled) setDatabaseContext(payload);
      })
      .catch((error: unknown) => {
        if (!cancelled) setContextError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) setContextLoading(false);
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
  const stages = stagesFromContext(databaseContext);
  const samplingConfig = databaseContext?.context?.sampling_config;
  const extraContext = remainingStructuredContext(databaseContext);
  const stopReasons = stages.map((stage) => stage.stopReason).filter(Boolean);
  return (
    <div className="reference-modal-backdrop" onClick={onClose}>
      <div className="reference-modal" onClick={(event) => event.stopPropagation()}>
        <header>
          <strong>完整模型上下文</strong>
          <button type="button" onClick={onClose}>关闭</button>
        </header>
        <div className="reference-modal-grid">
          <div className="context-main">
            {contextError ? <div className="error-bar">context 加载失败：{contextError}</div> : null}
            {contextLoading
              ? <div className="spinner">正在读取完整模型上下文…</div>
              : null}
            {databaseContext?.errors.map((error, index) => (
              <div className="error-bar" key={`${index}:${error}`}>context 解析提示：{error}</div>
            ))}
            {stages.map((stage, index) => (
              <div className="context-stage" key={index}>
                <ContextBlock label={`stage ${index + 1} · assembled prompt`} value={stage.prompt || "—"} />
                <ContextBlock
                  label={`stage ${index + 1} · raw completion${stage.stopReason ? ` · stop=${stage.stopReason}` : ""}`}
                  value={stage.completion || "—"}
                />
              </div>
            ))}
            {databaseContext?.view === "text" && databaseContext.raw_text ? (
              <ContextBlock label="raw context" value={databaseContext.raw_text} />
            ) : null}
            {databaseContext?.view === "structured" && stages.length === 0 && databaseContext.context ? (
              <ContextBlock label="structured context" value={readableValue(databaseContext.context)} />
            ) : null}
            {extraContext ? <ContextBlock label="additional context" value={readableValue(extraContext)} /> : null}
            {contextError ? (
              <>
                <ContextBlock label="context preview · fallback" value={answer.prompt} />
                <ContextBlock label="extracted answer · fallback" value={answer.completion} />
              </>
            ) : null}
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
              <span className="success">准确率：{accuracy.toFixed(1)}%</span>
            </div>
            <dl className="context-key-values">
              <dt>problem_id</dt><dd>{answer.id}</dd>
              <dt>repeat_id</dt><dd>{answer.repeatId}</dd>
              <dt>pass_index</dt><dd>{answer.passIndex}</dd>
              <dt>task_id</dt><dd>{taskId}</dd>
              <dt>architecture</dt><dd>{architectureLabel(selection.architecture)}</dd>
              <dt>is_passed</dt><dd><span className={`answer-outcome ${passed ? "pass" : "fail"}`}>{passed ? "true" : "false"}</span></dd>
              <dt>final_stop_reason</dt><dd>{answer.source.final_stop_reason || "—"}</dd>
              <dt>final_truncated</dt><dd>{answer.source.is_truncated ? "true" : "false"}</dd>
            </dl>
            {samplingConfig !== undefined ? (
              <ContextBlock label="sampling_config" value={readableValue(samplingConfig) || "{}"} />
            ) : null}
            <h3>SCORING RESULT</h3>
            <dl className="context-key-values">
              <dt>ground_truth</dt><dd>{answer.groundTruth}</dd>
              <dt>extracted_answer</dt><dd>{answer.modelAnswer}</dd>
              <dt>fail_reason</dt><dd>{answer.failReason || "—"}</dd>
            </dl>
            <h3>CONTEXT METADATA</h3>
            <dl className="context-key-values">
              <dt>context_view</dt><dd>{databaseContext?.view ?? (contextLoading ? "loading" : "unavailable")}</dd>
              <dt>stage_count</dt><dd>{stages.length}</dd>
              <dt>stop_reason</dt><dd>{stopReasons.join(", ") || "—"}</dd>
              <dt>context_errors</dt><dd>{databaseContext?.errors.length ?? 0}</dd>
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
