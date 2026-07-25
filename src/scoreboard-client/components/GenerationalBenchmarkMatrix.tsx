"use client";

import type { CSSProperties } from "react";
import { useMemo, useState } from "react";

import type {
  LeaderboardMatrix,
  MatrixCell,
  MatrixColumn,
  MatrixDomain,
  MatrixRow,
} from "../lib/dtos/api/leaderboard";
import { pct } from "./format";

type ParameterGroup = {
  param: string;
  current: MatrixRow;
  previous: MatrixRow | null;
};

type Comparison = {
  group: ParameterGroup;
  current: MatrixCell;
  previous: MatrixCell | null;
  delta: number | null;
};

type BenchmarkRow = {
  column: MatrixColumn;
  comparisons: Comparison[];
  improvementCount: number;
  regressionCount: number;
  weaknessCount: number;
  missingCount: number;
  severity: number;
};

type DiagnosticItem = {
  benchmark: string;
  value: number;
};

type ParameterDiagnostic = {
  param: string;
  weakest: DiagnosticItem[];
  regressions: DiagnosticItem[];
  missing: number;
};

type ViewMode = "all" | "improvements" | "regressions" | "weaknesses" | "missing";
type SortMode = "issues" | "benchmark";

const VIEW_LABELS: Record<ViewMode, string> = {
  all: "全部",
  improvements: "提升",
  regressions: "回退",
  weaknesses: "弱项",
  missing: "缺失",
};

function generationTimestamp(model: string): number {
  const matches = [...model.matchAll(/20\d{6}/g)];
  return Number(matches.at(-1)?.[0] ?? 0);
}

function parameterValue(param: string): number {
  return Number.parseFloat(param.replace(/[^\d.]/g, "")) || 0;
}

function groupsForDomain(domain: MatrixDomain | undefined): ParameterGroup[] {
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
    .sort((left, right) => parameterValue(right.param) - parameterValue(left.param));
}

function ScoreValue({
  cell,
  tone = "current",
}: {
  cell: MatrixCell | null;
  tone?: "current" | "previous" | "weak" | "regression";
}) {
  if (cell?.percent == null) {
    return <span className="generation-score missing">—</span>;
  }
  const width = Math.max(0, Math.min(100, cell.percent));
  const potential = cell.potential_percent == null
    ? null
    : Math.max(width, Math.min(100, cell.potential_percent));
  const style = {
    "--score-width": `${width}%`,
    "--potential-width": `${potential ?? width}%`,
  } as CSSProperties;
  const title = potential == null
    ? `标准分 ${pct(cell.percent)} · ${cell.metric ?? "score"}`
    : `标准分 ${pct(cell.percent)} · 潜力分 ${pct(potential)} · ${cell.metric ?? "score"}：k 次采样中至少一次答对`;
  return (
    <span className={`generation-score ${tone} ${potential == null ? "" : "has-potential"}`} style={style} title={title}>
      <span className="score-numbers">
        <strong>{cell.percent.toFixed(1)}</strong>
        {potential == null
          ? <em>%</em>
          : <em title={`括号内为潜力分 ${pct(potential)}`}> ({potential.toFixed(1)})%</em>}
      </span>
      <i aria-hidden="true">
        {potential == null ? null : <b className="potential-bar" />}
        <b className="standard-bar" />
      </i>
    </span>
  );
}

function DeltaValue({ value }: { value: number | null }) {
  if (value == null) return <span className="generation-delta neutral">—</span>;
  const className = value > 0.05 ? "positive" : value < -0.05 ? "negative" : "neutral";
  return (
    <span className={`generation-delta ${className}`}>
      {value > 0 ? "+" : ""}{value.toFixed(1)}
    </span>
  );
}

function AverageDelta({ group }: { group: ParameterGroup }) {
  const current = group.current.average;
  const previous = group.previous?.average ?? null;
  const delta = current == null || previous == null ? null : current - previous;
  return (
    <span className="group-average">
      {pct(current)} / {pct(previous)} <DeltaValue value={delta} />
    </span>
  );
}

export function GenerationalBenchmarkMatrix({ matrix }: { matrix: LeaderboardMatrix }) {
  const initialDomain = matrix.domains.find((item) => item.columns.length) ?? matrix.domains[0];
  const [domainKey, setDomainKey] = useState(initialDomain?.key ?? "knowledge");
  const [query, setQuery] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("all");
  const [sortMode, setSortMode] = useState<SortMode>("issues");
  const domain = matrix.domains.find((item) => item.key === domainKey) ?? initialDomain;
  const groups = useMemo(() => groupsForDomain(domain), [domain]);

  const diagnostics = useMemo<ParameterDiagnostic[]>(() => {
    if (!domain) return [];
    return groups.map((group) => {
      const values = domain.columns.map((column, index) => {
        const current = group.current.cells[index]?.percent ?? null;
        const previous = group.previous?.cells[index]?.percent ?? null;
        return {
          benchmark: column.label,
          current,
          delta: current == null || previous == null ? null : current - previous,
        };
      });
      return {
        param: group.param,
        weakest: values
          .filter((item): item is typeof item & { current: number } => item.current != null)
          .sort((left, right) => left.current - right.current)
          .slice(0, 3)
          .map((item) => ({ benchmark: item.benchmark, value: item.current })),
        regressions: values
          .filter((item): item is typeof item & { delta: number } => item.delta != null && item.delta < -0.05)
          .sort((left, right) => left.delta - right.delta)
          .slice(0, 3)
          .map((item) => ({ benchmark: item.benchmark, value: item.delta })),
        missing: values.filter((item) => item.current == null).length,
      };
    });
  }, [domain, groups]);

  const allRows = useMemo(() => {
    if (!domain) return [];
    return domain.columns.map((column, index): BenchmarkRow => {
      const comparisons = groups.map((group) => {
        const current = group.current.cells[index];
        const previous = group.previous?.cells[index] ?? null;
        return {
          group,
          current,
          previous,
          delta: current?.percent == null || previous?.percent == null
            ? null
            : current.percent - previous.percent,
        };
      });
      const weaknessCount = comparisons.filter((comparison) =>
        diagnostics
          .find((item) => item.param === comparison.group.param)
          ?.weakest.some((item) => item.benchmark === column.label),
      ).length;
      const regressionDeltas = comparisons
        .map((comparison) => comparison.delta)
        .filter((delta): delta is number => delta != null && delta < -0.05);
      const improvementCount = comparisons.filter(
        (comparison) => comparison.delta != null && comparison.delta > 0.05,
      ).length;
      const missingCount = comparisons.filter(
        (comparison) => comparison.current?.percent == null || comparison.previous?.percent == null,
      ).length;
      return {
        column,
        comparisons,
        improvementCount,
        regressionCount: regressionDeltas.length,
        weaknessCount,
        missingCount,
        severity: regressionDeltas.reduce((sum, delta) => sum + Math.abs(delta), 0)
          + weaknessCount * 5
          + missingCount * 8,
      };
    });
  }, [diagnostics, domain, groups]);

  const issueCounts = useMemo<Record<ViewMode, number>>(() => ({
    all: allRows.length,
    improvements: allRows.filter((row) => row.improvementCount).length,
    regressions: allRows.filter((row) => row.regressionCount).length,
    weaknesses: allRows.filter((row) => row.weaknessCount).length,
    missing: allRows.filter((row) => row.missingCount).length,
  }), [allRows]);

  const rows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return allRows
      .filter((row) => !normalizedQuery || row.column.label.toLowerCase().includes(normalizedQuery))
      .filter((row) => {
        if (viewMode === "improvements") return row.improvementCount > 0;
        if (viewMode === "regressions") return row.regressionCount > 0;
        if (viewMode === "weaknesses") return row.weaknessCount > 0;
        if (viewMode === "missing") return row.missingCount > 0;
        return true;
      })
      .sort((left, right) => (
        sortMode === "issues"
          ? right.severity - left.severity || left.column.label.localeCompare(right.column.label)
          : left.column.label.localeCompare(right.column.label)
      ));
  }, [allRows, query, sortMode, viewMode]);

  if (!domain || !groups.length) {
    return <div className="empty">暂无模拟评测数据。</div>;
  }

  return (
    <section className="generation-board">
      <div className="generation-toolbar">
        <label className="benchmark-search">
          <span>Benchmark</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索 Benchmark"
          />
          <strong>{rows.length}/{domain.columns.length}</strong>
        </label>
        <nav className="matrix-domain-tabs generation-domain-tabs" aria-label="领域筛选">
          {matrix.domains.map((item) => (
            <button
              type="button"
              key={item.key}
              className={item.key === domain.key ? "active" : ""}
              disabled={!item.columns.length}
              onClick={() => {
                setDomainKey(item.key);
                setQuery("");
                setViewMode("all");
                setSortMode("issues");
              }}
            >
              {item.label}<span>{item.columns.length}</span>
            </button>
          ))}
        </nav>
        <div className="generation-toolbar-meta">
          <span>{groups.length} 参数量</span>
          <span>{groups.length * 2} 模型</span>
          <span className="mock-data-badge">SIMULATED DATA</span>
        </div>
      </div>

      <div className="generation-context">
        <div className="evaluation-purpose">
          <strong>{domain.title}</strong>
          <span>当前表现 / 代际变化 / 真实弱项与缺失</span>
        </div>
        <span className="generation-legend">
          <i className="current-dot" /> 当前代
          <i className="previous-dot" /> 上一代
          <i className="progress-sample"><b /></i> 标准分
          <i className="progress-sample potential-sample"><b /></i> 潜力分（至少一次答对）
        </span>
      </div>

      <div className="issue-toolbar">
        <div className="issue-view-tabs" role="group" aria-label="评测结论筛选">
          {(Object.keys(VIEW_LABELS) as ViewMode[]).map((mode) => (
            <button
              type="button"
              key={mode}
              className={`${mode} ${viewMode === mode ? "active" : ""}`}
              onClick={() => setViewMode(mode)}
            >
              {VIEW_LABELS[mode]} <strong>{issueCounts[mode]}</strong>
            </button>
          ))}
        </div>
        <div className="parameter-issue-summary">
          {diagnostics.map((diagnostic) => (
            <span
              key={diagnostic.param}
              title={[
                `弱项: ${diagnostic.weakest.map((item) => `${item.benchmark} ${item.value.toFixed(1)}`).join(", ") || "无"}`,
                `回退: ${diagnostic.regressions.map((item) => `${item.benchmark} ${item.value.toFixed(1)}`).join(", ") || "无"}`,
                `缺失: ${diagnostic.missing}`,
              ].join("\n")}
            >
              <b>{diagnostic.param}</b>
              <i className="weak">弱{diagnostic.weakest.length}</i>
              <i className="regression">退{diagnostic.regressions.length}</i>
              {diagnostic.missing ? <i className="missing">缺{diagnostic.missing}</i> : null}
            </span>
          ))}
        </div>
        <button
          className="issue-sort"
          type="button"
          onClick={() => setSortMode((value) => value === "issues" ? "benchmark" : "issues")}
        >
          排序：{sortMode === "issues" ? "缺陷优先" : "Benchmark"}
        </button>
      </div>

      <div className="generation-table-wrap">
        <table className="generation-table generation-table-all-params">
          <thead>
            <tr className="parameter-header-row">
              <th className="benchmark-column" rowSpan={2}>Benchmark</th>
              <th className="signal-column" rowSpan={2}>
                <span>信号</span>
                <small>升/退/弱/缺</small>
              </th>
              <th className="metric-column" rowSpan={2}>Method<br />Metric</th>
              <th className="sample-column" rowSpan={2}>Samples</th>
              {groups.map((group) => (
                <th className="parameter-group-header" colSpan={3} key={group.param}>
                  <div>
                    <strong>{group.param}</strong>
                    <AverageDelta group={group} />
                  </div>
                </th>
              ))}
            </tr>
            <tr className="model-header-row">
              {groups.flatMap((group) => [
                <th className="model-score-column current-model group-start" key={`${group.param}:current`}>
                  <span className="generation-label current">CURRENT</span>
                  <strong title={group.current.model}>{group.current.model}</strong>
                </th>,
                <th className="model-score-column previous-model" key={`${group.param}:previous`}>
                  <span className="generation-label previous">PREVIOUS</span>
                  <strong title={group.previous?.model ?? ""}>{group.previous?.model ?? "无上一代"}</strong>
                </th>,
                <th className="delta-column" key={`${group.param}:delta`}>Δ</th>,
              ])}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const signalTitle = row.comparisons.map((comparison) => {
                const weak = diagnostics
                  .find((item) => item.param === comparison.group.param)
                  ?.weakest.some((item) => item.benchmark === row.column.label);
                const delta = comparison.delta == null
                  ? "不可比较"
                  : `${comparison.delta > 0 ? "+" : ""}${comparison.delta.toFixed(1)}`;
                return `${comparison.group.param}: Δ${delta}${weak ? " · 弱项" : ""}`;
              }).join("\n");
              return (
                <tr key={row.column.key}>
                  <td className="benchmark-column">
                    <strong title={row.column.label}>{row.column.label}</strong>
                  </td>
                  <td className="signal-column" title={signalTitle}>
                    <span className="signal-set">
                      {row.improvementCount ? <b className="improvement">↑{row.improvementCount}</b> : null}
                      {row.regressionCount ? <b className="regression">↓{row.regressionCount}</b> : null}
                      {row.weaknessCount ? <b className="weak">弱{row.weaknessCount}</b> : null}
                      {row.missingCount ? <b className="missing">缺{row.missingCount}</b> : null}
                      {!row.improvementCount && !row.regressionCount && !row.weaknessCount && !row.missingCount
                        ? <b className="neutral">—</b>
                        : null}
                    </span>
                  </td>
                  <td className="metric-column">
                    <span>{row.column.eval_method}</span>
                    <small>{row.column.metric ?? "score"}</small>
                  </td>
                  <td className="sample-column">{row.column.num_samples?.toLocaleString() ?? "—"}</td>
                  {row.comparisons.flatMap((comparison) => [
                    <td className="model-score-column current-model group-start" key={`${comparison.group.param}:current`}>
                      <ScoreValue
                        cell={comparison.current}
                        tone={comparison.delta != null && comparison.delta < -0.05
                          ? "regression"
                          : diagnostics
                            .find((item) => item.param === comparison.group.param)
                            ?.weakest.some((item) => item.benchmark === row.column.label)
                            ? "weak"
                            : "current"}
                      />
                    </td>,
                    <td className="model-score-column previous-model" key={`${comparison.group.param}:previous`}>
                      <ScoreValue cell={comparison.previous} tone="previous" />
                    </td>,
                    <td className="delta-column" key={`${comparison.group.param}:delta`}>
                      <DeltaValue value={comparison.delta} />
                    </td>,
                  ])}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
