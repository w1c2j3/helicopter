"use client";

import { useState } from "react";

import { api } from "../lib/api";
import type { LeaderboardResponse } from "../lib/dtos/api/leaderboard";
import type { MetaResponse } from "../lib/dtos/api/meta";
import { CapabilityMatrix } from "./MatrixScoreboard";

interface Props {
  meta: MetaResponse;
  leaderboard: LeaderboardResponse;
  model: string;
  view: string;
  tab: string;
}

export function DashboardPage({ meta, leaderboard }: Props) {
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const refresh = async () => {
    setRefreshError(null);
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
          <span>NAIVE MODEL LEADERBOARD</span>
          <h2>模型 × Benchmark 热力榜</h2>
          <p>保留领域导航；每个领域内纵向列出完整模型名，横向展示该领域全部 Benchmark。</p>
        </div>
        <button className="btn" type="button" onClick={refresh}>刷新数据</button>
      </div>
      <CapabilityMatrix matrix={leaderboard.matrix} />
    </>
  );
}
