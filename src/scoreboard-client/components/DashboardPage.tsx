"use client";

import { useState } from "react";

import { api } from "../lib/api";
import type { LeaderboardResponse } from "../lib/dtos/api/leaderboard";
import type { MetaResponse } from "../lib/dtos/api/meta";
import { GenerationalBenchmarkMatrix } from "./GenerationalBenchmarkMatrix";

interface Props {
  meta: MetaResponse;
  leaderboard: LeaderboardResponse;
  model: string;
  view: string;
  tab: string;
  isMockData: boolean;
}

export function DashboardPage({ meta, leaderboard, isMockData }: Props) {
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const refresh = async () => {
    setRefreshError(null);
    if (isMockData) {
      window.location.reload();
      return;
    }
    try {
      await api.refresh();
      window.location.reload();
    } catch (error) {
      setRefreshError(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <>
      {meta.errors.length ? <div className="error-bar">{meta.errors.join("; ")}</div> : null}
      {leaderboard.errors.length ? <div className="error-bar">{leaderboard.errors.join("; ")}</div> : null}
      {refreshError ? <div className="error-bar">刷新失败：{refreshError}</div> : null}
      <div className="matrix-page-heading">
        <div>
          <span>GENERATION BENCHMARK MATRIX</span>
          <h2>同参数量 · 当前一代 vs 上一代</h2>
          <p>Benchmark 纵向排列，模型横向排列；不同参数量完全隔离，不进行无意义的跨尺寸排名。</p>
        </div>
        <button className="btn" type="button" onClick={refresh}>
          {isMockData ? "刷新模拟数据" : "刷新数据"}
        </button>
      </div>
      <GenerationalBenchmarkMatrix matrix={leaderboard.matrix} />
    </>
  );
}
