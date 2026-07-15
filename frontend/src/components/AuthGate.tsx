"use client";

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api, ApiRequestError } from "@/lib/api";
import { AUTH_EXPIRED_EVENT } from "@/lib/auth-events";
import type { AppUser } from "@/types";

interface AuthGateProps {
  children: (user: AppUser, logout: () => void) => ReactNode;
}

const SESSION_RESTORE_TIMEOUT_MS = 15_000;
const USER_STORAGE_KEY = "vllm_manager_user";

function readStoredUser(): AppUser | null {
  try {
    const raw = window.localStorage.getItem(USER_STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as AppUser;
  } catch {
    return null;
  }
}

function persistUser(user: AppUser | null) {
  if (!user) {
    window.localStorage.removeItem(USER_STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
}

function shouldClearStoredToken(error: unknown): boolean {
  if (error instanceof ApiRequestError) {
    return error.status === 401 || error.status === 403;
  }
  return false;
}

export default function AuthGate({ children }: AuthGateProps) {
  const [user, setUser] = useState<AppUser | null>(null);
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(true);
  const [loggingIn, setLoggingIn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newPassword, setNewPassword] = useState("");
  const [newPasswordConfirm, setNewPasswordConfirm] = useState("");
  const [changingPassword, setChangingPassword] = useState(false);
  const [changePasswordError, setChangePasswordError] = useState<string | null>(null);

  useEffect(() => {
    function onAuthExpired() {
      window.localStorage.removeItem("vllm_manager_token");
      persistUser(null);
      setUser(null);
      setError("セッションが失効しました。再ログインしてください。");
    }
    window.addEventListener(AUTH_EXPIRED_EVENT, onAuthExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onAuthExpired);
  }, []);

  useEffect(() => {
    async function restore() {
      const token = window.localStorage.getItem("vllm_manager_token");
      if (!token) {
        setLoading(false);
        return;
      }
      const cachedUser = readStoredUser();
      if (cachedUser) {
        setUser(cachedUser);
        setLoading(false);
      }
      try {
        setError(null);
        const restored = await Promise.race([
          api.me(),
          new Promise<never>((_, reject) => {
            window.setTimeout(() => {
              reject(
                new ApiRequestError(
                  "セッション復元がタイムアウトしました。バックエンドの起動を待ってから再読み込みするか、再ログインしてください。"
                )
              );
            }, SESSION_RESTORE_TIMEOUT_MS);
          }),
        ]);
        setUser(restored);
        persistUser(restored);
      } catch (e) {
        if (shouldClearStoredToken(e)) {
          window.localStorage.removeItem("vllm_manager_token");
          persistUser(null);
          setUser(null);
        } else if (!cachedUser) {
          const msg =
            e instanceof ApiRequestError
              ? e.message
              : e instanceof Error
                ? e.message
                : String(e);
          setError(msg);
        }
      } finally {
        if (!cachedUser) {
          setLoading(false);
        }
      }
    }
    restore();
  }, []);

  async function login() {
    setError(null);
    setLoggingIn(true);
    try {
      const result = await api.login({ username, password });
      window.localStorage.setItem("vllm_manager_token", result.token);
      persistUser(result.user);
      setUser(result.user);
    } catch (err) {
      setError(err instanceof ApiRequestError ? err.message : String(err));
    } finally {
      setLoggingIn(false);
    }
  }

  function logout() {
    window.localStorage.removeItem("vllm_manager_token");
    persistUser(null);
    setUser(null);
  }

  async function submitPasswordChange() {
    setChangePasswordError(null);
    if (newPassword.length < 8) {
      setChangePasswordError("パスワードは8文字以上にしてください。");
      return;
    }
    if (newPassword !== newPasswordConfirm) {
      setChangePasswordError("確認用パスワードが一致しません。");
      return;
    }
    setChangingPassword(true);
    try {
      const updated = await api.updateMe({ password: newPassword });
      persistUser(updated);
      setUser(updated);
      setNewPassword("");
      setNewPasswordConfirm("");
    } catch (err) {
      setChangePasswordError(err instanceof ApiRequestError ? err.message : String(err));
    } finally {
      setChangingPassword(false);
    }
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
              disabled={loggingIn}
              className="w-full px-4 py-2 bg-accent-primary text-white rounded-lg font-medium disabled:opacity-50"
            >
              {loggingIn ? "ログイン中..." : "ログイン"}
            </button>
            <p className="text-xs text-gray-500">
              初期値は環境変数 `VLLM_MANAGER_ADMIN_USER` / `VLLM_MANAGER_ADMIN_PASSWORD` で変更できます。
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (user.must_change_password) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg-primary px-4">
        <div className="w-full max-w-sm bg-bg-secondary rounded-xl border border-accent-warning/40 p-6">
          <h1 className="text-xl font-bold mb-1">パスワードの変更が必要です</h1>
          <p className="text-sm text-gray-400 mb-6">
            {user.username} のパスワードは推測されやすい既定値のままです。続行する前に新しいパスワードを設定してください。
          </p>
          <div className="space-y-3">
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="新しいパスワード（8文字以上）"
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white"
            />
            <input
              type="password"
              value={newPasswordConfirm}
              onChange={(e) => setNewPasswordConfirm(e.target.value)}
              placeholder="新しいパスワード（確認）"
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white"
              onKeyDown={(e) => {
                if (e.key === "Enter") submitPasswordChange();
              }}
            />
            {changePasswordError && (
              <p className="text-sm text-accent-danger">{changePasswordError}</p>
            )}
            <button
              onClick={submitPasswordChange}
              disabled={changingPassword}
              className="w-full px-4 py-2 bg-accent-primary text-white rounded-lg font-medium disabled:opacity-50"
            >
              {changingPassword ? "変更中..." : "パスワードを変更して続ける"}
            </button>
            <button
              onClick={logout}
              className="w-full text-xs text-gray-500 hover:text-white"
            >
              ログアウト
            </button>
          </div>
        </div>
      </div>
    );
  }

  return <>{children(user, logout)}</>;
}
