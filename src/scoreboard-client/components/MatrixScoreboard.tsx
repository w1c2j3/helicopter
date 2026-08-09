"use client";

import type { CSSProperties } from "react";
import { useMemo, useState } from "react";

import type {
  CellMeta,
  LeaderboardMatrix,
  MatrixCell,
  TuningBenchmark,
  TuningMatrix,
} from "../lib/dtos/api/leaderboard";
import { EvalRecordsPanel } from "./EvalRecordsPanel";
import { pct } from "./format";

type SortDirection = "asc" | "desc";

function ScoreHeat({ cell, onSelect }: { cell: MatrixCell; onSelect: (meta: CellMeta) => void }) {
  if (cell.percent == null) return <span className="heat-missing">—</span>;
  const width = Math.max(0, Math.min(100, cell.percent));
  const hue = Math.round(18 + width * 1.05);
  const style = {
    "--heat-width": `${width}%`,
    "--heat-color": `hsl(${hue} 62% 26%)`,
  } as CSSProperties;
  return (
    <button
      className={`heat-cell${cell.meta?.clickable ? " clickable" : ""}`}
      style={style}
      type="button"
      disabled={!cell.meta?.clickable}
      title={`${cell.metric ?? "score"} · ${cell.num_samples ?? "?"} samples`}
      onClick={() => cell.meta?.clickable && onSelect(cell.meta)}
    >
      <span>{pct(cell.percent)}</span>
    </button>
  );
}

function ModelSearch({ value, onChange, visible, total }: {
  value: string;
  onChange: (value: string) => void;
  visible: number;
  total: number;
}) {
  return (
    <div className="matrix-search">
      <span aria-hidden="true">⌕</span>
      <input
        aria-label="搜索模型"
        type="search"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Search by full model / weight name"
      />
      <strong>{visible} / {total}</strong>
      {value ? <button type="button" onClick={() => onChange("")}>清除</button> : null}
    </div>
  );
}

export function CapabilityMatrix({ matrix }: { matrix: LeaderboardMatrix }) {
  const initialDomain = matrix.domains.find((item) => item.key === "knowledge" && item.columns.length)
    ?? matrix.domains.find((item) => item.columns.length)
    ?? matrix.domains[0];
  const [domainKey, setDomainKey] = useState(initialDomain?.key ?? "knowledge");
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState("average");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [selectedMeta, setSelectedMeta] = useState<CellMeta | null>(null);
  const domain = matrix.domains.find((item) => item.key === domainKey) ?? initialDomain;
  const rows = useMemo(() => {
    if (!domain) return [];
    const normalized = query.trim().toLowerCase();
    const columnIndex = domain.columns.findIndex((column) => column.key === sortKey);
    return domain.rows
      .filter((row) => !normalized || row.model.toLowerCase().includes(normalized))
      .slice()
      .sort((left, right) => {
        const leftValue = sortKey === "average" ? left.average : left.cells[columnIndex]?.percent;
        const rightValue = sortKey === "average" ? right.average : right.cells[columnIndex]?.percent;
        if (leftValue == null && rightValue == null) return left.model.localeCompare(right.model);
        if (leftValue == null) return 1;
        if (rightValue == null) return -1;
        return (leftValue - rightValue) * (sortDirection === "asc" ? 1 : -1);
      });
  }, [domain, query, sortDirection, sortKey]);

  const sortBy = (key: string) => {
    if (sortKey === key) setSortDirection((current) => current === "desc" ? "asc" : "desc");
    else {
      setSortKey(key);
      setSortDirection("desc");
    }
  };

  if (!domain) return <div className="empty">暂无 Naive 评测数据。</div>;
  return (
    <>
      <section className="matrix-controls">
        <ModelSearch value={query} onChange={setQuery} visible={rows.length} total={domain.rows.length} />
        <nav className="matrix-domain-tabs" aria-label="领域筛选">
          {matrix.domains.map((item) => (
            <button
              type="button"
              key={item.key}
              className={item.key === domain.key ? "active" : ""}
              disabled={!item.columns.length}
              onClick={() => {
                setDomainKey(item.key);
                setSortKey("average");
                setSortDirection("desc");
              }}
            >
              {item.label}<span>{item.columns.length}</span>
            </button>
          ))}
        </nav>
      </section>

      <div className="matrix-summary">
        <div><strong>{domain.title}</strong><span>Naive scores · 模型为纵轴 · Benchmark 为横轴</span></div>
        <span>{domain.columns.length} Benchmarks · {rows.length} Models</span>
      </div>

      <div className="matrix-scroll-shell">
        <div className="matrix-scroll-hint">← 横向滚动查看该领域剩余 Benchmark →</div>
        <div className="matrix-scroll" data-testid="capability-matrix-scroll">
          <table className="heatmap-table">
            <thead>
              <tr>
                <th className="sticky-rank">Rank</th>
                <th className="sticky-param">Params</th>
                <th className="sticky-model">Model / Weight</th>
                <th className="heat-score heat-average">
                  <button type="button" onClick={() => sortBy("average")}>Average {sortKey === "average" ? (sortDirection === "desc" ? "↓" : "↑") : ""}</button>
                </th>
                {domain.columns.map((column) => (
                  <th className="heat-score" key={column.key}>
                    <button type="button" onClick={() => sortBy(column.key)}>{column.label} {sortKey === column.key ? (sortDirection === "desc" ? "↓" : "↑") : ""}</button>
                    <small>{column.eval_method} · {column.metric ?? "score"}</small>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={row.model}>
                  <td className="sticky-rank"><span className={index < 3 ? `matrix-rank rank-${index + 1}` : ""}>{index + 1}</span></td>
                  <td className="sticky-param">{row.param}</td>
                  <td className="sticky-model"><strong>{row.model}</strong><small>{row.coverage}/{domain.columns.length} completed</small></td>
                  <td className="heat-score heat-average"><ScoreHeat cell={{ percent: row.average, meta: null, metric: "average", num_samples: null, created_at: null }} onSelect={setSelectedMeta} /></td>
                  {row.cells.map((cell, cellIndex) => (
                    <td className="heat-score" key={domain.columns[cellIndex]?.key ?? cellIndex}><ScoreHeat cell={cell} onSelect={setSelectedMeta} /></td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 ? <div className="empty">没有匹配的模型。</div> : null}
        </div>
      </div>
      <EvalRecordsPanel meta={selectedMeta} onClose={() => setSelectedMeta(null)} />
    </>
  );
}

export function TuningMatrixBoard({ matrix }: { matrix: TuningMatrix }) {
  const [benchmarkKey, setBenchmarkKey] = useState(matrix.benchmarks[0]?.key ?? "");
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState("best");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [selectedMeta, setSelectedMeta] = useState<CellMeta | null>(null);
  const benchmark = matrix.benchmarks.find((item) => item.key === benchmarkKey) ?? matrix.benchmarks[0];
  const rows = useMemo(() => sortTuningRows(benchmark, query, sortKey, sortDirection), [benchmark, query, sortDirection, sortKey]);

  if (!benchmark) return <div className="empty">暂无 Normal / 参数搜索数据。</div>;
  const sortBy = (key: string) => {
    if (sortKey === key) setSortDirection((current) => current === "desc" ? "asc" : "desc");
    else {
      setSortKey(key);
      setSortDirection("desc");
    }
  };
  return (
    <>
      <section className="matrix-controls tuning-controls">
        <div className="benchmark-select">
          <label htmlFor="tuning-benchmark">Benchmark</label>
          <select id="tuning-benchmark" value={benchmark.key} onChange={(event) => { setBenchmarkKey(event.target.value); setSortKey("best"); }}>
            {matrix.benchmarks.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
          </select>
          <span>{benchmark.metric ?? "score"} · {benchmark.num_samples ?? "?"} samples</span>
        </div>
        <ModelSearch value={query} onChange={setQuery} visible={rows.length} total={benchmark.rows.length} />
      </section>
      <div className="matrix-summary">
        <div><strong>{benchmark.label}</strong><span>Normal / 参数搜索 · 权重为纵轴 · 提示词与采样配置为横轴</span></div>
        <span>{benchmark.columns.length} Configs · {rows.length} Weights</span>
      </div>
      <div className="matrix-scroll-shell">
        <div className="matrix-scroll-hint">← 横向滚动查看剩余提示词与采样配置 →</div>
        <div className="matrix-scroll">
          <table className="heatmap-table tuning-heatmap">
            <thead>
              <tr>
                <th className="sticky-rank">Rank</th>
                <th className="sticky-param">Params</th>
                <th className="sticky-model">Model / Weight</th>
                <th className="heat-score heat-average"><button type="button" onClick={() => sortBy("best")}>Best {sortKey === "best" ? (sortDirection === "desc" ? "↓" : "↑") : ""}</button></th>
                {benchmark.columns.map((column) => (
                  <th className="heat-score tuning-config" key={column.key} title={JSON.stringify(column.sampling_config, null, 2)}>
                    <button type="button" onClick={() => sortBy(column.key)}>{column.label} {sortKey === column.key ? (sortDirection === "desc" ? "↓" : "↑") : ""}</button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={row.model}>
                  <td className="sticky-rank"><span className={index < 3 ? `matrix-rank rank-${index + 1}` : ""}>{index + 1}</span></td>
                  <td className="sticky-param">{row.param}</td>
                  <td className="sticky-model"><strong>{row.model}</strong><small>{row.coverage}/{benchmark.columns.length} configs</small></td>
                  <td className="heat-score heat-average"><ScoreHeat cell={{ percent: row.best, meta: null, metric: "best", num_samples: null, created_at: null }} onSelect={setSelectedMeta} /></td>
                  {row.cells.map((cell, cellIndex) => (
                    <td className="heat-score" key={benchmark.columns[cellIndex]?.key ?? cellIndex}><ScoreHeat cell={cell} onSelect={setSelectedMeta} /></td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <EvalRecordsPanel meta={selectedMeta} onClose={() => setSelectedMeta(null)} />
    </>
  );
}

function sortTuningRows(benchmark: TuningBenchmark | undefined, query: string, sortKey: string, direction: SortDirection) {
  if (!benchmark) return [];
  const normalized = query.trim().toLowerCase();
  const columnIndex = benchmark.columns.findIndex((column) => column.key === sortKey);
  return benchmark.rows
    .filter((row) => !normalized || row.model.toLowerCase().includes(normalized))
    .slice()
    .sort((left, right) => {
      const leftValue = sortKey === "best" ? left.best : left.cells[columnIndex]?.percent;
      const rightValue = sortKey === "best" ? right.best : right.cells[columnIndex]?.percent;
      if (leftValue == null && rightValue == null) return left.model.localeCompare(right.model);
      if (leftValue == null) return 1;
      if (rightValue == null) return -1;
      return (leftValue - rightValue) * (direction === "asc" ? 1 : -1);
    });
}
