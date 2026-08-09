const SERVER_API_BASE = process.env.SCOREBOARD_API_BASE_URL || "http://127.0.0.1:7860";
const BROWSER_API_BASE = process.env.NEXT_PUBLIC_SCOREBOARD_API_BASE_URL || "";
const GET_TIMEOUT_MS = 20_000;

function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (typeof window === "undefined") return `${SERVER_API_BASE}${normalized}`;
  return BROWSER_API_BASE ? `${BROWSER_API_BASE}${normalized}` : normalized;
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), {
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal: AbortSignal.timeout(GET_TIMEOUT_MS),
  });
  if (!res.ok) {
    throw new Error(`${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

function adminHeaders(extra?: Record<string, string>): Record<string, string> {
  return { Accept: "application/json", "X-RWKV-Admin-Request": "1", ...extra };
}

export async function getJsonAuth<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), {
    headers: adminHeaders(),
    cache: "no-store",
    credentials: "same-origin",
  });
  if (!res.ok) {
    throw new Error(`${res.status}: ${await errText(res)}`);
  }
  return res.json() as Promise<T>;
}

export async function postJsonAuth<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: adminHeaders({ "Content-Type": "application/json" }),
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: "same-origin",
  });
  if (!res.ok) {
    throw new Error(`${res.status}: ${await errText(res)}`);
  }
  return res.json() as Promise<T>;
}

async function errText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return typeof data?.detail === "string" ? data.detail : JSON.stringify(data);
  } catch {
    return res.statusText;
  }
}
