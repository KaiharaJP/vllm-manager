// API クライアント

import { notifyAuthExpired } from "@/lib/auth-events";

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
      if (resp.status === 401 && token) {
        notifyAuthExpired();
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
  // 永続APIトークン（PAT）: 自分自身
  getMyTokens: () =>
    request<{ tokens: import("@/types").ApiToken[]; count: number }>("/api/auth/me/tokens"),
  createMyToken: (body: { name: string; expires_in_days?: number }) =>
    request<import("@/types").ApiTokenCreated>("/api/auth/me/tokens", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  revokeMyToken: (tokenId: string) =>
    request<{ success: boolean; id: string }>(`/api/auth/me/tokens/${encodeURIComponent(tokenId)}`, {
      method: "DELETE",
    }),

  // 永続APIトークン（PAT）: 管理者による他ユーザー分の閲覧・強制失効
  getUserTokens: (username: string) =>
    request<{ tokens: import("@/types").ApiToken[]; count: number }>(
      `/api/users/${encodeURIComponent(username)}/tokens`
    ),
  revokeUserToken: (username: string, tokenId: string) =>
    request<{ success: boolean; id: string; username: string }>(
      `/api/users/${encodeURIComponent(username)}/tokens/${encodeURIComponent(tokenId)}`,
      { method: "DELETE" }
    ),

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
  getInstances: () => request<import("@/types").ServerInstance[]>("/api/instances"),
  stopInstance: (instance_id: string) =>
    request<{ success: boolean; message: string; instance_id?: string }>("/api/instances/stop", {
      method: "POST",
      body: JSON.stringify({ instance_id }),
    }),
  stopServerByPid: (pid: number) =>
    request<{ success: boolean; message: string; pid?: number }>("/api/servers/stop", {
      method: "POST",
      body: JSON.stringify({ pid }),
    }),
  restartServer: () =>
    request<import("@/types").ApiResponse>("/api/restart", { method: "POST" }),
  runSmokeTest: (instance_id: string) =>
    request<import("@/types").SmokeTestResult>(
      `/api/instances/${encodeURIComponent(instance_id)}/smoke-test`,
      { method: "POST" }
    ),

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
  resumeModelDownload: (model_id: string) =>
    request<import("@/types").DownloadJob>("/api/model-downloads/resume", {
      method: "POST",
      body: JSON.stringify({ model_id }),
    }),
  getContextPresets: () =>
    request<import("@/types").ContextPreset[]>("/api/context-presets"),
  getLog: (tail = 100, instanceId?: string) =>
    request<{ log: string }>(
      `/api/log?tail=${tail}${instanceId ? `&instance_id=${encodeURIComponent(instanceId)}` : ""}`
    ),
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

  // チャット UI（バックエンドプロキシ経由）
  getChatModels: () => request<import("@/types").ChatModelsResponse>("/api/chat/models"),
};

export interface StreamChatCompletionOptions {
  model: string;
  messages: import("@/types").ChatUiMessage[];
  temperature?: number;
  max_tokens?: number;
  signal?: AbortSignal;
  onDelta: (text: string) => void;
}

/** SSE ストリームでチャット補完を実行する（既存 request() は JSON のみ対応のため別実装） */
export async function streamChatCompletion(options: StreamChatCompletionOptions): Promise<void> {
  const url = API_BASE ? `${API_BASE}/api/chat/completions` : "/api/chat/completions";
  const token = getToken();
  const resp = await fetch(url, {
    method: "POST",
    signal: options.signal,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      model: options.model,
      messages: options.messages,
      stream: true,
      ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
      ...(options.max_tokens !== undefined ? { max_tokens: options.max_tokens } : {}),
    }),
  });

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
    if (resp.status === 401 && token) {
      notifyAuthExpired();
    }
    throw new ApiRequestError(`API Error: ${resp.status} ${detail}`, resp.status);
  }

  const reader = resp.body?.getReader();
  if (!reader) {
    throw new ApiRequestError("ストリーミング応答を読み取れませんでした");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      try {
        const parsed = JSON.parse(data) as {
          choices?: Array<{ delta?: { content?: string } }>;
        };
        const delta = parsed.choices?.[0]?.delta?.content;
        if (typeof delta === "string" && delta.length > 0) {
          options.onDelta(delta);
        }
      } catch {
        /* ignore malformed SSE chunks */
      }
    }
  }
}
