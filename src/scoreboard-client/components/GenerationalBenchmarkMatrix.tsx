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

function ScoreValue({ cell }: { cell: MatrixCell | null }) {
  if (cell?.percent == null) {
    return <span className="generation-score missing">—</span>;
  }
  const width = Math.max(0, Math.min(100, cell.percent));
  const style = { "--score-width": `${width}%` } as CSSProperties;
  return (
    <span className="generation-score" style={style}>
      <strong>{pct(cell.percent)}</strong>
      <i aria-hidden="true"><b /></i>
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
  const initialDomain = matrix.domains.find((domain) => domain.columns.length) ?? matrix.domains[0];
  const [domainKey, setDomainKey] = useState(initialDomain?.key ?? "knowledge");
  const [query, setQuery] = useState("");
  const [reverseBenchmarks, setReverseBenchmarks] = useState(false);
  const domain = matrix.domains.find((item) => item.key === domainKey) ?? initialDomain;
  const groups = useMemo(() => groupsForDomain(domain), [domain]);

  const rows = useMemo(() => {
    if (!domain) return [];
    const normalizedQuery = query.trim().toLowerCase();
    const result: BenchmarkRow[] = domain.columns
      .map((column, index) => ({
        column,
        comparisons: groups.map((group) => {
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
        }),
      }))
      .filter((row) => !normalizedQuery || row.column.label.toLowerCase().includes(normalizedQuery))
      .sort((left, right) => left.column.label.localeCompare(right.column.label));
    return reverseBenchmarks ? result.reverse() : result;
  }, [domain, groups, query, reverseBenchmarks]);

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
                setReverseBenchmarks(false);
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
        <div>
          <strong>{domain.title}</strong>
          <span>Benchmark 为行 · 四个参数量同表 · 每组仅比较当前代与上一代</span>
        </div>
        <span className="generation-legend">
          <i className="current-dot" /> 当前代
          <i className="previous-dot" /> 上一代
          <i className="progress-sample"><b /></i> 分数进度
        </span>
      </div>

      <div className="generation-table-wrap">
        <table className="generation-table generation-table-all-params">
          <thead>
            <tr className="parameter-header-row">
              <th className="benchmark-column" rowSpan={2}>
                <button type="button" onClick={() => setReverseBenchmarks((value) => !value)}>
                  Benchmark {reverseBenchmarks ? "↓" : "↑"}
                </button>
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
            {rows.map((row) => (
              <tr key={row.column.key}>
                <td className="benchmark-column"><strong>{row.column.label}</strong></td>
                <td className="metric-column">
                  <span>{row.column.eval_method}</span>
                  <small>{row.column.metric ?? "score"}</small>
                </td>
                <td className="sample-column">{row.column.num_samples?.toLocaleString() ?? "—"}</td>
                {row.comparisons.flatMap((comparison) => [
                  <td className="model-score-column current-model group-start" key={`${comparison.group.param}:current`}>
                    <ScoreValue cell={comparison.current} />
                  </td>,
                  <td className="model-score-column previous-model" key={`${comparison.group.param}:previous`}>
                    <ScoreValue cell={comparison.previous} />
                  </td>,
                  <td className="delta-column" key={`${comparison.group.param}:delta`}>
                    <DeltaValue value={comparison.delta} />
                  </td>,
                ])}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
