// API クライアント

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!resp.ok) {
    throw new Error(`API Error: ${resp.status} ${resp.statusText}`);
  }

  return resp.json();
}

export const api = {
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
  getContextPresets: () =>
    request<import("@/types").ContextPreset[]>("/api/context-presets"),
  getLog: (tail = 100) => request<{ log: string }>(`/api/log?tail=${tail}`),

  // LiteLLM
  getLiteLLMStatus: () =>
    request<import("@/types").LiteLLMStatus>("/api/litellm/status"),
};
