// API クライアント

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");

/**
 * fetch の中止だけでは TCP が固まった環境で効かないことがあるため、
 * 「fetch + JSON パース」全体を wall-clock で打ち切る。
 */
const REQUEST_DEADLINE_MS = 28_000;

export class ApiRequestError extends Error {
  declare readonly cause?: unknown;

  constructor(
    message: string,
    public readonly status?: number,
    options?: { cause?: unknown }
  ) {
    super(message);
    this.name = "ApiRequestError";
    if (options?.cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
  }
}

/** いずれかが abort したらもう一方も打ち切る */
function mergeAbortSignals(deadline: AbortSignal, outer?: AbortSignal | null): AbortSignal {
  if (!outer) return deadline;
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.any === "function") {
    return AbortSignal.any([deadline, outer]);
  }
  const ctrl = new AbortController();
  const abort = () => ctrl.abort();
  deadline.addEventListener("abort", abort);
  outer.addEventListener("abort", abort);
  return ctrl.signal;
}

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem("vllm_manager_token");
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = API_BASE ? `${API_BASE}${path}` : path;
  const backendLabel = API_BASE || "同一オリジン (/api 経由)";
  const token = getToken();
  const deadlineCtrl = new AbortController();
  const deadlineTimer = setTimeout(() => deadlineCtrl.abort(), REQUEST_DEADLINE_MS);
  const signal = mergeAbortSignals(deadlineCtrl.signal, options?.signal);

  try {
    let resp: Response;
    try {
      resp = await fetch(url, {
        ...options,
        signal,
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
          ...options?.headers,
        },
      });
    } catch (e) {
      const isAbort =
        e instanceof Error &&
        (e.name === "AbortError" || e.name === "TimeoutError");
      const msg = isAbort
        ? `API の応答がタイムアウトしました（${REQUEST_DEADLINE_MS / 1000}秒以内に完了しませんでした）。バックエンド (${backendLabel}) が起動しているか、ブラウザから到達できるか確認してください。`
        : `API に接続できませんでした: ${e instanceof Error ? e.message : String(e)}`;
      throw new ApiRequestError(msg, undefined, { cause: e });
    }

    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const body = await resp.json();
        if (body != null && typeof body === "object" && "detail" in body) {
          const d = (body as { detail: unknown }).detail;
          detail = typeof d === "string" ? d : JSON.stringify(d);
        }
      } catch {
        /* ignore */
      }
      throw new ApiRequestError(`API Error: ${resp.status} ${detail}`, resp.status);
    }

    return (await resp.json()) as T;
  } finally {
    clearTimeout(deadlineTimer);
  }
}

export const api = {
  // 認証
  login: (body: { username: string; password: string }) =>
    request<{ token: string; user: import("@/types").AppUser }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  me: () => request<import("@/types").AppUser>("/api/auth/me"),
  updateMe: (body: { password?: string; litellm_team_id?: string }) =>
    request<import("@/types").AppUser>("/api/auth/me", {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  getMyApiKeys: () => request<{ keys: unknown[]; count: number }>("/api/auth/me/api-keys"),
  createMyApiKey: (body: {
    models?: string[];
    max_budget?: number;
    budget_duration?: string;
    rpm_limit?: number;
    tpm_limit?: number;
  }) =>
    request<unknown>("/api/auth/me/api-keys", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getUserApiKeys: (username: string) =>
    request<{ keys: unknown[]; count: number; user_id?: string; team_id?: string | null }>(
      `/api/users/${encodeURIComponent(username)}/api-keys`
    ),
  createUserApiKey: (
    username: string,
    body: {
      models?: string[];
      max_budget?: number;
      budget_duration?: string;
      rpm_limit?: number;
      tpm_limit?: number;
    }
  ) =>
    request<unknown>(`/api/users/${encodeURIComponent(username)}/api-keys`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getUsers: () => request<import("@/types").AppUser[]>("/api/users"),
  createUser: (body: {
    username: string;
    password: string;
    role: "admin" | "user";
    litellm_user_id?: string;
    litellm_team_id?: string;
  }) =>
    request<import("@/types").AppUser>("/api/users", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateUser: (username: string, body: {
    password?: string;
    role?: "admin" | "user";
    litellm_user_id?: string;
    litellm_team_id?: string;
    disabled?: boolean;
  }) =>
    request<import("@/types").AppUser>(`/api/users/${encodeURIComponent(username)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  // サーバー管理
  getStatus: () => request<import("@/types").ServerStatus>("/api/status"),
  checkServiceHealth: () =>
    request<import("@/types").ServiceHealthCheckResult>("/api/health/check"),
  startServer: (body: import("@/types").ServerStartRequest) =>
    request<import("@/types").ApiResponse>("/api/start", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  stopServer: () =>
    request<import("@/types").ApiResponse>("/api/stop", { method: "POST" }),
  getRunningServers: () => request<import("@/types").RunningServer[]>("/api/servers"),
  stopServerByPid: (pid: number) =>
    request<{ success: boolean; message: string; pid?: number }>("/api/servers/stop", {
      method: "POST",
      body: JSON.stringify({ pid }),
    }),
  restartServer: () =>
    request<import("@/types").ApiResponse>("/api/restart", { method: "POST" }),

  // 設定・モデル
  getConfig: () => request<import("@/types").ServerConfig>("/api/config"),
  getModels: () => request<import("@/types").Model[]>("/api/models"),
  registerModel: (body: Partial<import("@/types").Model> & { id: string }) =>
    request<import("@/types").Model>("/api/models", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteModel: (model_id: string) =>
    request<{
      model_id: string;
      removed: boolean;
      removed_entries: number;
      cache_deleted: boolean;
      cache_deleted_paths: string[];
      bytes_freed: number;
    }>(
      `/api/models/${encodeURIComponent(model_id)}`,
      {
        method: "DELETE",
      }
    ),
  deleteModelCache: (model_id: string) =>
    request<{ model_id: string; deleted: boolean; deleted_paths: string[]; bytes_freed: number }>(
      `/api/models/${encodeURIComponent(model_id)}/cache`,
      {
        method: "DELETE",
      }
    ),
  getModelDownloads: () =>
    request<import("@/types").DownloadJob[]>("/api/model-downloads"),
  startModelDownload: (model_id: string, force = false) =>
    request<import("@/types").DownloadJob>("/api/model-downloads", {
      method: "POST",
      body: JSON.stringify({ model_id, force }),
    }),
  cancelModelDownloads: (model_id: string) =>
    request<{ model_id: string; cancelled_count: number; cancelled_job_ids: string[] }>(
      "/api/model-downloads/cancel",
      {
        method: "POST",
        body: JSON.stringify({ model_id }),
      }
    ),
  getContextPresets: () =>
    request<import("@/types").ContextPreset[]>("/api/context-presets"),
  getLog: (tail = 100) => request<{ log: string }>(`/api/log?tail=${tail}`),
  getSystemMetrics: () => request<import("@/types").SystemMetrics>("/api/system-metrics"),

  // LiteLLM
  getLiteLLMStatus: () =>
    request<import("@/types").LiteLLMStatus>("/api/litellm/status"),
  getLiteLLMKeys: () => request<unknown>("/api/litellm/keys"),
  createLiteLLMKey: (payload: Record<string, unknown>) =>
    request<unknown>("/api/litellm/keys", {
      method: "POST",
      body: JSON.stringify({ payload }),
    }),
  deleteLiteLLMKey: (payload: Record<string, unknown>) =>
    request<unknown>("/api/litellm/keys/delete", {
      method: "POST",
      body: JSON.stringify({ payload }),
    }),
  getLiteLLMUsers: () => request<unknown>("/api/litellm/users"),
  createLiteLLMUser: (payload: Record<string, unknown>) =>
    request<unknown>("/api/litellm/users", {
      method: "POST",
      body: JSON.stringify({ payload }),
    }),
  getLiteLLMTeams: () => request<unknown>("/api/litellm/teams"),
  createLiteLLMTeam: (payload: Record<string, unknown>) =>
    request<unknown>("/api/litellm/teams", {
      method: "POST",
      body: JSON.stringify({ payload }),
    }),
  getLiteLLMSpend: () => request<unknown>("/api/litellm/spend"),

  // モニタリング / リクエスト履歴（admin）
  getLiteLLMProxyRequestDetail: (trackId: string) =>
    request<import("@/types").LiteLLMProxyRequestDetail>(
      `/api/admin/litellm-proxy-requests/${encodeURIComponent(trackId)}`
    ),
  getRequestHistory: (limit = 50, offset = 0) =>
    request<import("@/types").RequestHistoryListResponse>(
      `/api/admin/request-history?limit=${limit}&offset=${offset}`
    ),
  getRequestHistoryDetail: (recordId: string) =>
    request<import("@/types").LiteLLMProxyRequestDetail>(
      `/api/admin/request-history/${encodeURIComponent(recordId)}`
    ),
};
