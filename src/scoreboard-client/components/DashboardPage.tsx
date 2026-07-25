"use client";

import type { LeaderboardResponse } from "../lib/dtos/api/leaderboard";
import type { MetaResponse } from "../lib/dtos/api/meta";
import { ReferenceEvaluationBoard } from "./ReferenceEvaluationBoard";

interface Props {
  meta: MetaResponse;
  leaderboard: LeaderboardResponse;
  model: string;
  view: string;
  tab: string;
  isMockData: boolean;
}

export function DashboardPage({ meta, leaderboard }: Props) {
  return (
    <>
      {meta.errors.length ? <div className="error-bar">{meta.errors.join("; ")}</div> : null}
      {leaderboard.errors.length ? <div className="error-bar">{leaderboard.errors.join("; ")}</div> : null}
      <ReferenceEvaluationBoard matrix={leaderboard.matrix} />
    </>
  );
}
