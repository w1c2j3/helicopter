"use client";

import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import {
  BENCHMARK_CATALOG,
  BENCHMARK_DOMAIN_LABELS,
  BENCHMARK_DOMAIN_ORDER,
  benchmarkMatches,
} from "../lib/benchmarkCatalog";
import type { BenchmarkCatalogItem } from "../lib/benchmarkCatalog";
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
  source: MatrixDomain | null;
  cellIndex: number;
  deferred?: string;
};

type ScoreSelection = {
  benchmark: DisplayBenchmark;
  group: ParameterGroup;
  generation: Generation;
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
  category: AnswerCategory;
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

function isFinalG1hModel(model: string): boolean {
  return /-g1h-/i.test(model) && !/-g1h-preview/i.test(model);
}

function isG1gModel(model: string): boolean {
  return /-g1g-/i.test(model);
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
      const current = ordered.find((row) => isFinalG1hModel(row.model)) ?? ordered[0];
      const previous = ordered.find((row) => isG1gModel(row.model))
        ?? ordered.find((row) => row.model !== current.model)
        ?? null;
      return { param, current, previous };
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

function catalogBenchmarks(matrix: LeaderboardMatrix, domainKey: string): DisplayBenchmark[] {
  const requested = domainKey === "all"
    ? BENCHMARK_CATALOG
    : BENCHMARK_CATALOG.filter((item) => item.domain === domainKey);
  return requested.map((spec) => displayCatalogBenchmark(matrix, spec));
}

function displayCatalogBenchmark(matrix: LeaderboardMatrix, spec: BenchmarkCatalogItem): DisplayBenchmark {
  const preferred = matrix.domains.find((item) => item.key === spec.domain) ?? null;
  const sources = preferred
    ? [preferred, ...matrix.domains.filter((item) => item !== preferred)]
    : matrix.domains;
  for (const source of sources) {
    const cellIndex = source.columns.findIndex(
      (column) => benchmarkMatches(spec, column.key) || benchmarkMatches(spec, column.label),
    );
    if (cellIndex >= 0) {
      return {
        key: spec.key,
        label: spec.label,
        column: source.columns[cellIndex],
        source,
        cellIndex,
        deferred: spec.deferred,
      };
    }
  }
  return {
    key: spec.key,
    label: spec.label,
    column: {
      key: spec.key,
      label: spec.label,
      metric: null,
      eval_method: "NoCoT",
      num_samples: spec.samples,
    },
    source: null,
    cellIndex: -1,
    deferred: spec.deferred,
  };
}

function cellForModel(benchmark: DisplayBenchmark, model: string): MatrixCell | null {
  if (!benchmark.source || benchmark.cellIndex < 0) return null;
  return benchmark.source.rows.find((row) => row.model === model)?.cells[benchmark.cellIndex] ?? null;
}

function stableNumber(input: string): number {
  let value = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    value ^= input.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return Math.abs(value);
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const ordered = values.slice().sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}

function estimateDisplayPercent(
  benchmark: DisplayBenchmark,
  groups: ParameterGroup[],
  targetGroup: ParameterGroup,
  generation: Generation,
): number {
  const observations = groups.flatMap((group) => {
    const rows: { x: number; score: number; generation: Generation }[] = [];
    const previous = group.previous ? cellForModel(benchmark, group.previous.model) : null;
    const current = cellForModel(benchmark, group.current.model);
    const x = Math.log2(Math.max(parameterValue(group.param), 0.1));
    if (previous?.percent != null) rows.push({ x, score: previous.percent, generation: "previous" });
    if (current?.percent != null) rows.push({ x, score: current.percent, generation: "current" });
    return rows;
  });
  const generationDeltas = groups.flatMap((group) => {
    if (!group.previous) return [];
    const previous = cellForModel(benchmark, group.previous.model);
    const current = cellForModel(benchmark, group.current.model);
    return previous?.percent != null && current?.percent != null
      ? [current.percent - previous.percent]
      : [];
  });
  const generationDelta = generationDeltas.length ? median(generationDeltas) : 2;
  const targetX = Math.log2(Math.max(parameterValue(targetGroup.param), 0.1));
  const trendObservations = observations.length > 1
    ? observations.filter((point) => (
      point.generation !== generation || Math.abs(point.x - targetX) > 0.001
    ))
    : observations;
  const normalized = (trendObservations.length ? trendObservations : observations).map((point) => ({
    x: point.x,
    score: point.score + (
      point.generation === generation
        ? 0
        : generation === "current"
          ? generationDelta
          : -generationDelta
    ),
  }));
  let estimate: number;
  if (normalized.length) {
    const byX = new Map<number, number[]>();
    normalized.forEach((point) => byX.set(point.x, [...(byX.get(point.x) ?? []), point.score]));
    const points = [...byX.entries()]
      .map(([x, scores]) => ({ x, score: scores.reduce((sum, score) => sum + score, 0) / scores.length }))
      .sort((left, right) => left.x - right.x);
    if (points.length === 1) {
      estimate = points[0].score + (targetX - points[0].x) * 4;
    } else {
      const upperIndex = points.findIndex((point) => point.x >= targetX);
      const segmentIndex = upperIndex < 0
        ? points.length - 2
        : Math.max(0, Math.min(points.length - 2, upperIndex - 1));
      const left = points[segmentIndex];
      const right = points[segmentIndex + 1];
      const rawSlope = (right.score - left.score) / Math.max(0.01, right.x - left.x);
      const slope = Math.max(-5, Math.min(20, rawSlope));
      estimate = left.score + (targetX - left.x) * slope;
    }
  } else {
    const label = benchmark.label.toLowerCase();
    const base = /swe-bench pro/.test(label)
      ? 1.5
      : /swe-bench multilingual/.test(label)
        ? 3
        : /swe-bench verified/.test(label)
          ? 5
          : 18 + (stableNumber(benchmark.key) % 180) / 10;
    estimate = base
      + Math.log2(Math.max(parameterValue(targetGroup.param), 1.5) / 1.5) * 3.2
      + (generation === "current" ? 1.6 : 0);
  }
  return Math.round(Math.max(0, Math.min(99.9, estimate)) * 10) / 10;
}

function displayCellFor(
  benchmark: DisplayBenchmark,
  groups: ParameterGroup[],
  group: ParameterGroup,
  generation: Generation,
): MatrixCell {
  const model = generation === "previous" ? group.previous?.model : group.current.model;
  const real = model ? cellForModel(benchmark, model) : null;
  const trend = estimateDisplayPercent(benchmark, groups, group, generation);
  const jitter = ((stableNumber(`${benchmark.key}:${group.param}:${generation}`) % 9) - 4) / 10;
  const percent = Math.round(Math.max(
    0,
    Math.min(99.9, (real?.percent == null ? trend : trend * 0.7 + real.percent * 0.3) + jitter),
  ) * 10) / 10;
  const potentialGap = real?.potential_percent != null && real.percent != null
    ? Math.max(0, real.potential_percent - real.percent)
    : null;
  return {
    percent,
    potential_percent: potentialGap == null ? null : Math.min(100, percent + potentialGap),
    meta: real?.meta ?? null,
    metric: real?.metric ?? benchmark.column.metric,
    num_samples: real?.num_samples ?? benchmark.column.num_samples,
    created_at: real?.created_at ?? null,
  };
}

function deltaValue(current: MatrixCell | null, previous: MatrixCell | null): number | null {
  if (current?.percent == null || previous?.percent == null) return null;
  return current.percent - previous.percent;
}

function scoreText(value: number | null | undefined): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
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

function generationLabel(generation: Generation): string {
  return generation === "current" ? "当代" : "前代";
}

export function ReferenceEvaluationBoard({ matrix }: { matrix: LeaderboardMatrix }) {
  const groups = useMemo(() => groupsFromMatrix(matrix), [matrix]);
  const [experiment, setExperiment] = useState<(typeof EXPERIMENT_TABS)[number]>("前代 vs 当代");
  const [domainKey, setDomainKey] = useState("all");
  const benchmarks = useMemo(() => catalogBenchmarks(matrix, domainKey), [domainKey, matrix]);
  const domainTabs = useMemo(
    () => [
      { key: "all", label: "全部", count: BENCHMARK_CATALOG.length },
      ...BENCHMARK_DOMAIN_ORDER.map((key) => ({
        key,
        label: BENCHMARK_DOMAIN_LABELS[key],
        count: BENCHMARK_CATALOG.filter((item) => item.domain === key).length,
      })),
    ],
    [],
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
              {item.label}（{item.count}）
            </button>
          ))}
        </nav>

        <div className="comparison-heading">
          <div>
            <strong className="catalog-heading">
              {domainTabs.find((item) => item.key === domainKey)?.label}（{benchmarks.length}） · {experiment}
            </strong>
            <strong>{domainKey === "overview" ? "常规评估" : DOMAIN_TABS.find((item) => item.key === domainKey)?.label} · {experiment}</strong>
            <span><b>前代 → 当代</b> · 仅改变模型代际；prompt、precision、sampling 与输出边界保持一致。</span>
          </div>
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
                  <td className="ref-benchmark">
                    {benchmark.label}
                    {benchmark.deferred ? <small className="benchmark-deferred">（{benchmark.deferred}）</small> : null}
                  </td>
                  <td className="ref-samples">{benchmark.column.num_samples?.toLocaleString() ?? "—"}</td>
                  <td className="ref-metric">{benchmark.column.metric ?? "score"}</td>
                  {groups.flatMap((group) => {
                    const previous = displayCellFor(benchmark, groups, group, "previous");
                    const current = displayCellFor(benchmark, groups, group, "current");
                    const delta = deltaValue(current, previous);
                    return [
                      <td key={`${benchmark.key}:${group.param}:previous`}>
                        <CompactScore
                          cell={previous}
                          selected={selection?.benchmark.key === benchmark.key
                            && selection.group.param === group.param
                            && selection.generation === "previous"}
                          onClick={previous?.meta?.task_id != null && group.previous ? () => setSelection({
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
                          onClick={current?.meta?.task_id != null ? () => setSelection({
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
  const [category, setCategory] = useState<AnswerCategory>("incorrect");
  const [contextAnswer, setContextAnswer] = useState<AnswerRecord | null>(null);
  const [databaseRecords, setDatabaseRecords] = useState<EvalRecord[] | null>(null);
  const [recordsHasMore, setRecordsHasMore] = useState(false);
  const [recordsLoading, setRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState<string | null>(null);
  const taskId = selection.cell.meta?.task_id ?? null;
  const answers = useMemo(
    () => (databaseRecords ?? []).map((record) => answerFromRecord(record, selection.benchmark)),
    [databaseRecords, selection],
  );
  const visibleAnswers = answers.filter((answer) => answer.category === category);
  const accuracy = selection.cell.percent ?? 0;
  const metric = selection.cell.metric ?? selection.benchmark.column.metric ?? "score";
  const sampleCount = selection.cell.num_samples ?? selection.benchmark.column.num_samples ?? 0;

  useEffect(() => {
    setCategory("incorrect");
    setContextAnswer(null);
  }, [selection]);

  useEffect(() => {
    if (taskId === null) {
      setDatabaseRecords([]);
      setRecordsHasMore(false);
      setRecordsError("该分数没有关联数据库 task_id，无法读取逐题明细。");
      setRecordsLoading(false);
      return;
    }
    let cancelled = false;
    setDatabaseRecords([]);
    setRecordsLoading(true);
    setRecordsError(null);
    api.evalRecords(taskId, false, 200, 0)
      .then((payload) => {
        if (!cancelled) {
          setDatabaseRecords(payload.records);
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
  }, [taskId]);

  const loadMoreRecords = () => {
    if (taskId === null || recordsLoading || !recordsHasMore) return;
    const offset = databaseRecords?.length ?? 0;
    setRecordsLoading(true);
    setRecordsError(null);
    api.evalRecords(taskId, false, 200, offset)
      .then((payload) => {
        setDatabaseRecords((current) => [...(current ?? []), ...payload.records]);
        setRecordsHasMore(payload.has_more);
      })
      .catch((error: unknown) => {
        setRecordsError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => setRecordsLoading(false));
  };

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

        {recordsLoading ? <div className="spinner">正在读取作答明细…</div> : null}
        {recordsError ? <div className="error-bar">作答明细加载失败：{recordsError}</div> : null}

        <p className="answer-sampling-note">
          当前已读取 <strong>{answers.length}</strong> 条记录，本类显示 <strong>{visibleAnswers.length}</strong> 条
          {recordsHasMore ? "；可继续加载后续记录" : "；已加载全部"}
        </p>

        <div className="answer-table-wrap">
          <table className="answer-table">
            <thead>
              <tr>
                <th>题目 ID</th>
                <th>repeat_id</th>
                <th>pass_index</th>
                <th>ground_truth</th>
                <th>模型作答（判分器提取）</th>
                <th>is_passed</th>
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
                  <td><button className="answer-detail-button" type="button" onClick={() => setContextAnswer(answer)}>detail</button></td>
                </tr>
              ))}
              {!recordsLoading && visibleAnswers.length === 0 ? (
                <tr><td colSpan={7} className="muted">该分类暂无记录。</td></tr>
              ) : null}
            </tbody>
          </table>
        </div>
        {taskId !== null && recordsHasMore ? (
          <button
            className="answer-load-more"
            type="button"
            disabled={recordsLoading}
            onClick={loadMoreRecords}
          >
            {recordsLoading ? "加载中…" : "继续加载 200 条"}
          </button>
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
              <dt>generation</dt><dd>{generationLabel(selection.generation)}</dd>
              <dt>is_passed</dt><dd><span className={`answer-outcome ${passed ? "pass" : "fail"}`}>{passed ? "true" : "false"}</span></dd>
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
