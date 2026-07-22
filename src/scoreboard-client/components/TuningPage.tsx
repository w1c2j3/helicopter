import type { LeaderboardResponse } from "../lib/dtos/api/leaderboard";
import { TuningMatrixBoard } from "./MatrixScoreboard";

export function TuningPage({ leaderboard }: { leaderboard: LeaderboardResponse }) {
  return (
    <>
      <div className="matrix-page-heading">
        <div>
          <span>NORMAL / PARAMETER SEARCH</span>
          <h2>刷榜配置热力矩阵</h2>
          <p>选择 Benchmark，纵向比较完整权重名称，横向比较提示词和采样配置。</p>
        </div>
      </div>
      <TuningMatrixBoard matrix={leaderboard.tuning_matrix} />
    </>
  );
}
