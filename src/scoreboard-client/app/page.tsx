import { AdminPage } from "../components/AdminPage";
import { DashboardPage } from "../components/DashboardPage";
import { HistoryPage } from "../components/HistoryPage";
import { api } from "../lib/api";

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
  const view = value(params, "view", "benchmark_detail_delta");
  const model = value(params, "model", "");
  const tab = value(params, "tab", "knowledge");

  const isNormalBoard = page === "normal" || page === "history";
  const isDashboard = !isNormalBoard && page !== "admin";
  let loadError: string | null = null;
  const meta = isDashboard ? await api.meta().catch((error: unknown) => {
    loadError = error instanceof Error ? error.message : String(error);
    return null;
  }) : null;
  const selectedModel = meta ? model || meta.auto_label : model;
  const leaderboard = meta ? await api.leaderboard(selectedModel, view).catch((error: unknown) => {
    loadError = error instanceof Error ? error.message : String(error);
    return null;
  }) : null;

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1><span className="brand-dot">⦿</span> RWKV Skills</h1>
          <div className="subtitle">
            {isNormalBoard ? "Normal 刷榜 · 提示词与解码参数对比" : page === "admin" ? "评测调度与运行管理" : `评测看板 · ${leaderboard?.view_label ?? view}`}
          </div>
        </div>
        <nav className="page-nav">
          <a className={page === "dashboard" ? "active" : ""} href={pageHref(`/?page=dashboard&view=${view}&model=${encodeURIComponent(selectedModel)}&tab=${tab}`)}>
            评测看板
          </a>
          <a className={isNormalBoard ? "active" : ""} href={pageHref("/?page=normal")}>
            Normal 刷榜
          </a>
          <a className={page === "admin" ? "active" : ""} href={pageHref("/?page=admin")}>
            管理面板
          </a>
        </nav>
      </header>
      {loadError ? <div className="error-bar">加载评测看板失败：{loadError}</div> : null}
      {isNormalBoard ? (
        <HistoryPage />
      ) : page === "admin" ? (
        <AdminPage />
      ) : meta && leaderboard ? (
        <DashboardPage meta={meta} leaderboard={leaderboard} model={selectedModel} view={view} tab={tab} />
      ) : (
        <div className="empty">暂无数据。</div>
      )}
    </main>
  );
}
