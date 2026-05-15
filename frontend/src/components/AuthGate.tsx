"use client";

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api, ApiRequestError } from "@/lib/api";
import type { AppUser } from "@/types";

interface AuthGateProps {
  children: (user: AppUser, logout: () => void) => ReactNode;
}

export default function AuthGate({ children }: AuthGateProps) {
  const [user, setUser] = useState<AppUser | null>(null);
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function restore() {
      try {
        setError(null);
        if (window.localStorage.getItem("vllm_manager_token")) {
          const restored = await Promise.race([
            api.me(),
            new Promise<never>((_, reject) => {
              window.setTimeout(() => {
                reject(
                  new ApiRequestError(
                    "セッション復元がタイムアウトしました。再ログインしてください。"
                  )
                );
              }, 12_000);
            }),
          ]);
          setUser(restored);
        }
      } catch (e) {
        const status = e instanceof ApiRequestError ? e.status : undefined;
        if (status === 401 || status === 403 || e instanceof ApiRequestError) {
          window.localStorage.removeItem("vllm_manager_token");
        }
        const msg =
          e instanceof ApiRequestError
            ? e.message
            : e instanceof Error
              ? e.message
              : String(e);
        setError(msg);
      } finally {
        setLoading(false);
      }
    }
    restore();
  }, []);

  /** AbortSignal が効かず fetch が固まる環境向けの最終手段 */
  useEffect(() => {
    if (!loading) return;
    const t = window.setTimeout(() => setLoading(false), 32_000);
    return () => window.clearTimeout(t);
  }, [loading]);

  async function login() {
    setError(null);
    setLoading(true);
    try {
      const result = await api.login({ username, password });
      window.localStorage.setItem("vllm_manager_token", result.token);
      setUser(result.user);
    } catch (err) {
      setError(err instanceof ApiRequestError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  function logout() {
    window.localStorage.removeItem("vllm_manager_token");
    setUser(null);
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-gray-400">
        Loading...
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg-primary px-4">
        <div className="w-full max-w-sm bg-bg-secondary rounded-xl border border-white/5 p-6">
          <h1 className="text-xl font-bold mb-1">vLLM Manager</h1>
          <p className="text-sm text-gray-500 mb-6">管理画面にログインしてください</p>
          <div className="space-y-3">
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="username"
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white"
            />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="password"
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white"
              onKeyDown={(e) => {
                if (e.key === "Enter") login();
              }}
            />
            {error && <p className="text-sm text-accent-danger">{error}</p>}
            <button
              onClick={login}
              className="w-full px-4 py-2 bg-accent-primary text-white rounded-lg font-medium"
            >
              ログイン
            </button>
            <p className="text-xs text-gray-500">
              初期値は環境変数 `VLLM_MANAGER_ADMIN_USER` / `VLLM_MANAGER_ADMIN_PASSWORD` で変更できます。
            </p>
          </div>
        </div>
      </div>
    );
  }

  return <>{children(user, logout)}</>;
}
