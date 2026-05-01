// API クライアント

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem("vllm_manager_token");
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getToken();
  const resp = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!resp.ok) {
    throw new Error(`API Error: ${resp.status} ${resp.statusText}`);
  }

  return resp.json();
}

export const api = {
  // 認証
  login: (body: { username: string; password: string }) =>
    request<{ token: string; user: import("@/types").AppUser }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  me: () => request<import("@/types").AppUser>("/api/auth/me"),
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

  // サーバー管理
  getStatus: () => request<import("@/types").ServerStatus>("/api/status"),
  startServer: (body: import("@/types").ServerStartRequest) =>
    request<import("@/types").ApiResponse>("/api/start", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  stopServer: () =>
    request<import("@/types").ApiResponse>("/api/stop", { method: "POST" }),
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
  getModelDownloads: () =>
    request<import("@/types").DownloadJob[]>("/api/model-downloads"),
  startModelDownload: (model_id: string) =>
    request<import("@/types").DownloadJob>("/api/model-downloads", {
      method: "POST",
      body: JSON.stringify({ model_id }),
    }),
  getContextPresets: () =>
    request<import("@/types").ContextPreset[]>("/api/context-presets"),
  getLog: (tail = 100) => request<{ log: string }>(`/api/log?tail=${tail}`),

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
};
