import { AdminPage } from "../components/AdminPage";
import { DashboardPage } from "../components/DashboardPage";
import { HistoryPage } from "../components/HistoryPage";
import { TuningPage } from "../components/TuningPage";
import { api } from "../lib/api";
import { createMockLeaderboard, createMockMeta } from "../lib/mockLeaderboard";

export const dynamic = "force-dynamic";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;
const PAGE_BASE = normalizeBasePath(process.env.NEXT_PUBLIC_BASE_PATH);

function normalizeBasePath(input: string | undefined): string {
  const value = (input || "").trim();
  if (!value) return "";
  return `${value.startsWith("/") ? "" : "/"}${value}`.replace(/\/+$/, "");
}

function pageHref(path: string): string {
  return PAGE_BASE ? `${PAGE_BASE}${path}` : path;
}

function value(params: Record<string, string | string[] | undefined>, key: string, fallback: string): string {
  const raw = params[key];
  if (Array.isArray(raw)) return raw[0] || fallback;
  return raw || fallback;
}

export default async function Home({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const page = value(params, "page", "dashboard");
  const view = value(params, "view", "benchmark_detail_latest");
  const model = value(params, "model", "");
  const tab = value(params, "tab", "knowledge");
  const isHistory = page === "history";
  const isTuning = page === "normal" || page === "tuning";
  const isDashboard = !isHistory && !isTuning && page !== "admin";
  const needsLeaderboard = isDashboard || isTuning;
  const useMockData = process.env.SCOREBOARD_USE_MOCK_DATA === "true";
  let loadError: string | null = null;
  const meta = needsLeaderboard
    ? useMockData
      ? createMockMeta()
      : await api.meta().catch((error: unknown) => {
          loadError = error instanceof Error ? error.message : String(error);
          return null;
        })
    : null;
  const selectedModel = meta ? model || meta.auto_label : model;
  const leaderboard = meta
    ? useMockData
      ? createMockLeaderboard()
      : await api.leaderboard(selectedModel, view).catch((error: unknown) => {
          loadError = error instanceof Error ? error.message : String(error);
          return null;
        })
    : null;

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1><span className="brand-dot">⦿</span> RWKV Skills</h1>
          <div className="subtitle">
            {isTuning ? "Normal 刷榜 · 提示词与采样配置" : isHistory ? "历史运行与成绩来源" : page === "admin" ? "评测调度与运行管理" : "Naive 模型能力与 Benchmark 表现"}
          </div>
        </div>
        <nav className="page-nav">
          <a className={isDashboard ? "active" : ""} href={pageHref("/?page=dashboard")}>评测看板</a>
          <a className={isTuning ? "active" : ""} href={pageHref("/?page=tuning")}>Normal 刷榜</a>
          <a className={page === "admin" ? "active" : ""} href={pageHref("/?page=admin")}>管理面板</a>
        </nav>
      </header>
      {loadError ? <div className="error-bar">加载评测看板失败：{loadError}</div> : null}
      {isHistory ? (
        <HistoryPage />
      ) : isTuning && leaderboard ? (
        <TuningPage leaderboard={leaderboard} />
      ) : page === "admin" ? (
        <AdminPage />
      ) : meta && leaderboard ? (
        <DashboardPage
          meta={meta}
          leaderboard={leaderboard}
          model={selectedModel}
          view={view}
          tab={tab}
          isMockData={useMockData}
        />
      ) : (
        <div className="empty">暂无数据。</div>
      )}
    </main>
  );
}
