"use client";

import { useEffect, useMemo, useState } from "react";
import type { AppUser } from "@/types";
import { ClipboardCopy, Key, RefreshCw, Users } from "lucide-react";
import { api } from "@/lib/api";

interface UserManagementPanelProps {
  currentUser: AppUser;
}

function copyTextWithFallback(text: string): boolean {
  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    textarea.style.pointerEvents = "none";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
}

export default function UserManagementPanel({ currentUser }: UserManagementPanelProps) {
  const [users, setUsers] = useState<AppUser[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [selectedUser, setSelectedUser] = useState<AppUser | null>(null);
  const [newUser, setNewUser] = useState({
    username: "",
    password: "",
    role: "user" as "admin" | "user",
    litellm_team_id: "",
  });
  const [editUser, setEditUser] = useState({
    role: "user" as "admin" | "user",
    litellm_user_id: "",
    litellm_team_id: "",
    disabled: false,
    newPassword: "",
  });
  const [liteKeys, setLiteKeys] = useState<unknown>(null);
  const [apiMessage, setApiMessage] = useState<string | null>(null);
  const [keyOptions, setKeyOptions] = useState({
    models: "vllm-local",
    max_budget: "",
    budget_duration: "",
    rpm_limit: "",
    tpm_limit: "",
  });
  const [copiedKeySig, setCopiedKeySig] = useState<string | null>(null);
  const [issuedSecretsByToken, setIssuedSecretsByToken] = useState<Record<string, string>>({});

  type KeyInfo = {
    key_name?: string | null;
    user_id?: string | null;
    team_id?: string | null;
    models?: string[] | null;
    max_budget?: number | null;
    budget_duration?: string | null;
    rpm_limit?: number | null;
    tpm_limit?: number | null;
    created_at?: string | null;
    expires?: string | null;
  };
  type KeyEntry = { key?: string; deleteKey?: string; info?: KeyInfo };

  function rememberIssuedSecret(created: unknown) {
    if (!created || typeof created !== "object") return;
    const obj = created as Record<string, unknown>;
    const key = typeof obj.key === "string" ? obj.key.trim() : "";
    const token = typeof obj.token === "string" ? obj.token.trim() : "";
    if (!key.startsWith("sk-") || !token) return;
    setIssuedSecretsByToken((prev) => ({ ...prev, [token]: key }));
  }

  function parseModels(raw: string): string[] {
    return raw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  useEffect(() => {
    if (currentUser.role === "admin") {
      refreshUsers();
      return;
    }
    setSelectedUser(currentUser);
    setEditUser({
      role: currentUser.role,
      litellm_user_id: currentUser.litellm_user_id || currentUser.username,
      litellm_team_id: currentUser.litellm_team_id || "",
      disabled: Boolean(currentUser.disabled),
      newPassword: "",
    });
    refreshMyKeys();
  }, []);

  async function refreshUsers() {
    try {
      const localUsers = await api.getUsers();
      setUsers(localUsers);
      if (!selectedUser && localUsers.length > 0) {
        handleSelectUser(localUsers[0]);
      } else if (selectedUser) {
        const updated = localUsers.find((u) => u.username === selectedUser.username);
        if (updated) handleSelectUser(updated);
      }
    } catch (err) {
      setMessage(String(err));
    }
  }

  async function refreshMyKeys() {
    try {
      const result = await api.getMyApiKeys();
      setLiteKeys(result.keys);
    } catch (err) {
      setApiMessage(String(err));
    }
  }

  async function createLocalUser() {
    setMessage(null);
    try {
      await api.createUser({
        ...newUser,
        litellm_user_id: newUser.username,
        litellm_team_id: newUser.litellm_team_id || undefined,
      });
      setMessage("ユーザーを作成しました");
      setNewUser({ username: "", password: "", role: "user", litellm_team_id: "" });
      refreshUsers();
    } catch (err) {
      setMessage(String(err));
    }
  }

  function handleSelectUser(user: AppUser) {
    setSelectedUser(user);
    setEditUser({
      role: user.role,
      litellm_user_id: user.litellm_user_id || user.username,
      litellm_team_id: user.litellm_team_id || "",
      disabled: Boolean(user.disabled),
      newPassword: "",
    });
  }

  async function saveUserChanges() {
    if (!selectedUser) return;
    setMessage(null);
    try {
      await api.updateUser(selectedUser.username, {
        role: editUser.role,
        litellm_user_id: editUser.litellm_user_id || undefined,
        litellm_team_id: editUser.litellm_team_id || undefined,
        disabled: editUser.disabled,
        password: editUser.newPassword || undefined,
      });
      setEditUser((prev) => ({ ...prev, newPassword: "" }));
      setMessage("ユーザー情報を更新しました");
      refreshUsers();
    } catch (err) {
      setMessage(String(err));
    }
  }

  async function refreshLiteKeys() {
    if (currentUser.role !== "admin") {
      await refreshMyKeys();
      return;
    }
    setApiMessage(null);
    try {
      if (!selectedUser) {
        setApiMessage("ユーザーが選択されていません。");
        return;
      }
      const keys = await api.getUserApiKeys(selectedUser.username);
      setLiteKeys(keys.keys);
    } catch (err) {
      setApiMessage(String(err));
    }
  }

  useEffect(() => {
    if (selectedUser) {
      refreshLiteKeys();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedUser?.username]);

  const filteredKeys = useMemo(() => {
    if (!liteKeys) return [];
    return Array.isArray(liteKeys) ? liteKeys : [];
  }, [liteKeys, selectedUser]);
  const normalizedModelSelection = useMemo(() => parseModels(keyOptions.models), [keyOptions.models]);
  const hasModelSelection = normalizedModelSelection.length > 0;

  /** LiteLLM の /key/info 応答はバージョンでトップレベル key の有無が違うため複数パスから拾う */
  function pickSecretString(item: Record<string, unknown>): string {
    const tryStr = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : "");
    const looksLikeVirtualKey = (v: string) => v.startsWith("sk-");
    const candidates: unknown[] = [
      item.key,
      item.token,
      item.litellm_key,
      item.api_key,
    ];
    const infoRaw = item.info;
    if (infoRaw && typeof infoRaw === "object") {
      const i = infoRaw as Record<string, unknown>;
      candidates.push(i.key, i.token, i.litellm_key, i.api_key);
    }
    for (const c of candidates) {
      const s = tryStr(c);
      // 実際に API 認証に使える virtual key のみ採用する
      if (s && looksLikeVirtualKey(s)) return s;
    }
    return "";
  }

  function asKeyEntries(items: unknown[]): KeyEntry[] {
    return items
      .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
      .map((item) => {
        const infoRaw = item.info;
        const directSecret = pickSecretString(item);
        const rawDeleteKey =
          typeof item.key === "string" && item.key.trim().length > 0
            ? item.key.trim()
            : undefined;
        const rememberedSecret =
          rawDeleteKey && issuedSecretsByToken[rawDeleteKey]
            ? issuedSecretsByToken[rawDeleteKey]
            : undefined;
        const secret = directSecret || rememberedSecret || "";
        return {
          key: secret || undefined,
          // key/list ではハッシュのみ返るため、削除時はこの識別子を使う
          deleteKey: rawDeleteKey,
          info: infoRaw && typeof infoRaw === "object" ? (infoRaw as KeyInfo) : undefined,
        };
      });
  }

  async function copyApiKeyMaterial(text: string, sig: string, isRealApiKey: boolean) {
    const t = text.trim();
    if (!t) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(t);
      } else if (!copyTextWithFallback(t)) {
        throw new Error("fallback failed");
      }
      setCopiedKeySig(sig);
      if (!isRealApiKey) {
        setApiMessage("表示値（key_name）をコピーしました。これは識別子であり、Bearer 認証には使えません。");
      }
      window.setTimeout(() => setCopiedKeySig(null), 2000);
    } catch {
      if (copyTextWithFallback(t)) {
        setCopiedKeySig(sig);
        if (!isRealApiKey) {
          setApiMessage("表示値（key_name）をコピーしました。これは識別子であり、Bearer 認証には使えません。");
        }
        window.setTimeout(() => setCopiedKeySig(null), 2000);
        return;
      }
      setApiMessage("クリップボードへのコピーに失敗しました（HTTP配信やブラウザ権限でブロックされる場合があります）。");
    }
  }

  function ApiKeyCopyBlock({
    entry,
    index,
    scope,
  }: {
    entry: KeyEntry;
    index: number;
    scope: "user" | "admin";
  }) {
    const secret =
      typeof entry.key === "string" && entry.key.length > 0 ? entry.key : "";
    const fallback = entry.info?.key_name?.trim() || ""; // key_name は表示用（認証には使えない）
    const copyTarget = secret || fallback;
    const hasRealApiKey = Boolean(secret);
    const sig = `${scope}-${index}-${copyTarget.slice(0, 24)}`;
    const display =
      secret || fallback || "—";
    const copied = copiedKeySig === sig;
    return (
      <div className="w-full rounded-lg border border-accent-primary/35 bg-bg-tertiary/70 p-3 space-y-2">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch sm:justify-between sm:gap-4">
          <div className="min-w-0 flex-1 space-y-1">
            <p className="text-[11px] font-medium text-gray-400">API キー</p>
            <code className="block break-all text-[11px] leading-relaxed text-gray-100">{display}</code>
          </div>
          <button
            type="button"
            disabled={!copyTarget}
            title={copyTarget ? "クリップボードにコピー" : "コピーできる値がありません"}
            onClick={() => copyApiKeyMaterial(copyTarget, sig, hasRealApiKey)}
            className="inline-flex w-full shrink-0 items-center justify-center gap-2 rounded-lg bg-accent-primary px-4 py-2.5 text-sm font-medium text-white shadow-sm hover:bg-accent-primary/90 disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto sm:self-start"
          >
            <ClipboardCopy className="h-4 w-4 shrink-0" />
            {copied ? "コピーしました" : hasRealApiKey ? "APIキーをコピー" : "表示値をコピー"}
          </button>
        </div>
        {!secret && (
          <p className="text-[10px] leading-snug text-gray-500">
            一覧 API ではシークレット全文が返らない場合があります。発行直後のメッセージや LiteLLM 管理画面で全文を確認してください。表示されている名前だけでは認証に使えないことがあります。
          </p>
        )}
      </div>
    );
  }

  async function deleteApiKey(keyId?: string) {
    if (!keyId) return;
    if (currentUser.role !== "admin") {
      setApiMessage("APIキーの削除は管理者のみ実行できます。");
      return;
    }
    setApiMessage("削除中...");
    try {
      await api.deleteLiteLLMKey({ keys: [keyId] });
      setIssuedSecretsByToken((prev) => {
        if (!(keyId in prev)) return prev;
        const next = { ...prev };
        delete next[keyId];
        return next;
      });
      setApiMessage("APIキーを削除しました。");
      await refreshLiteKeys();
    } catch (err) {
      setApiMessage(String(err));
    }
  }

  async function issueApiKeyForSelectedUser() {
    setApiMessage("発行中...");
    try {
      if (currentUser.role !== "admin") {
        const models = parseModels(keyOptions.models);
        if (!models.length) {
          setApiMessage("モデル名を1つ以上指定してください。`*` を指定すると全モデルアクセスになります。");
          return;
        }
        const payload: any = { models: models.length ? models : ["vllm-local"] };
        if (keyOptions.max_budget) payload.max_budget = Number(keyOptions.max_budget);
        if (keyOptions.budget_duration) payload.budget_duration = keyOptions.budget_duration;
        if (keyOptions.rpm_limit) payload.rpm_limit = Number(keyOptions.rpm_limit);
        if (keyOptions.tpm_limit) payload.tpm_limit = Number(keyOptions.tpm_limit);
        const created = await api.createMyApiKey(payload);
        rememberIssuedSecret(created);
        if (created && typeof created === "object" && "key" in (created as any)) {
          const k = (created as any).key;
          if (typeof k === "string") {
            setApiMessage(`あなた向けの LiteLLM API キーを発行しました。\n\n${k}`);
          } else {
            setApiMessage("あなた向けの LiteLLM API キーを発行しました");
          }
        } else {
          setApiMessage("あなた向けの LiteLLM API キーを発行しました");
        }
        await refreshMyKeys();
        return;
      }
      if (!selectedUser) {
        setApiMessage("ユーザーが選択されていません。");
        return;
      }
      const models = parseModels(keyOptions.models);
      if (!models.length) {
        setApiMessage("モデル名を1つ以上指定してください。`*` を指定すると全モデルアクセスになります。");
        return;
      }
      const payload: any = { models: models.length ? models : ["vllm-local"] };
      if (keyOptions.max_budget) payload.max_budget = Number(keyOptions.max_budget);
      if (keyOptions.budget_duration) payload.budget_duration = keyOptions.budget_duration;
      if (keyOptions.rpm_limit) payload.rpm_limit = Number(keyOptions.rpm_limit);
      if (keyOptions.tpm_limit) payload.tpm_limit = Number(keyOptions.tpm_limit);
      const created = await api.createUserApiKey(selectedUser.username, payload);
      rememberIssuedSecret(created);
      setApiMessage("このユーザー向けの LiteLLM API キーを発行しました");
      await refreshLiteKeys();
      // 返ってきた key があれば表示（コピーしやすい）
      if (created && typeof created === "object" && "key" in (created as any)) {
        const k = (created as any).key;
        if (typeof k === "string") setApiMessage(`このユーザー向けの LiteLLM API キーを発行しました。\n\n${k}`);
      }
    } catch (err) {
      setApiMessage(String(err));
    }
  }

  async function updateMyProfile() {
    setMessage(null);
    try {
      await api.updateMe({
        password: editUser.newPassword || undefined,
        litellm_team_id: editUser.litellm_team_id || undefined,
      });
      setEditUser((prev) => ({ ...prev, newPassword: "" }));
      setMessage("プロフィールを更新しました");
    } catch (err) {
      setMessage(String(err));
    }
  }

  if (currentUser.role !== "admin") {
    return (
      <div className="space-y-6 animate-slide-in">
        <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
          <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
            <Users className="w-5 h-5 text-accent-primary" />
            マイページ
          </h2>
          <p className="text-sm text-gray-400 mb-4">
            自分の team_id / パスワード更新と、推論 API キーの発行・確認ができます。
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-gray-300">ユーザー名</span>
              <input
                disabled
                className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-gray-400"
                value={currentUser.username}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-gray-300">権限</span>
              <input
                disabled
                className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-gray-400"
                value="一般ユーザー"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-gray-300">LiteLLM user_id</span>
              <input
                disabled
                className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-gray-400"
                value={editUser.litellm_user_id || currentUser.username}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-gray-300">team_id（任意）</span>
              <input
                className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
                value={editUser.litellm_team_id}
                onChange={(e) => setEditUser({ ...editUser, litellm_team_id: e.target.value })}
              />
            </label>
            <label className="block md:col-span-2">
              <span className="mb-1 block text-xs font-medium text-gray-300">
                新しいパスワード（空欄なら変更しない）
              </span>
              <input
                type="password"
                className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
                value={editUser.newPassword}
                onChange={(e) => setEditUser({ ...editUser, newPassword: e.target.value })}
              />
            </label>
          </div>
          <button
            onClick={updateMyProfile}
            className="mt-4 px-4 py-2 bg-accent-primary text-white rounded-lg"
          >
            プロフィールを更新
          </button>
          {message && <p className="mt-3 text-sm text-gray-400">{message}</p>}
        </div>

        <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-200 flex items-center gap-2">
              <Key className="w-4 h-4 text-accent-warning" />
              あなたの API キー
            </h3>
            <div className="flex gap-2">
              <button
                onClick={refreshMyKeys}
                className="inline-flex items-center gap-1 rounded-md border border-white/10 px-2 py-1 text-[11px] text-gray-300 hover:bg-bg-primary"
              >
                <RefreshCw className="w-3 h-3" />
                再読込
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-400 mb-3">
            一般ユーザーはこの画面から API キーを発行できません。必要な場合は管理者に発行を依頼してください。
          </p>
          {apiMessage && <p className="text-xs text-gray-400 mb-2">{apiMessage}</p>}
          <div className="space-y-2">
            {asKeyEntries(filteredKeys as unknown[]).length > 0 ? (
              asKeyEntries(filteredKeys as unknown[]).map((entry, idx) => (
                <div
                  key={entry.key ?? `k-${idx}`}
                  className="rounded-lg border border-white/10 bg-bg-primary/70 p-3 text-xs"
                >
                  <ApiKeyCopyBlock entry={entry} index={idx} scope="user" />
                  <div className="mt-3 grid grid-cols-1 border-t border-white/10 pt-3 md:grid-cols-2 gap-2 text-gray-300">
                    <p><span className="text-gray-500">user_id:</span> {entry.info?.user_id || "-"}</p>
                    <p><span className="text-gray-500">team_id:</span> {entry.info?.team_id || "-"}</p>
                    <p><span className="text-gray-500">models:</span> {(entry.info?.models || []).join(", ") || "-"}</p>
                    <p><span className="text-gray-500">作成:</span> {entry.info?.created_at || "-"}</p>
                    <p><span className="text-gray-500">有効期限:</span> {entry.info?.expires || "なし"}</p>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-500 text-xs">まだ発行済み API キーはありません。</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
          <Users className="w-5 h-5 text-accent-primary" />
          ユーザー管理
        </h2>
        <div className="mb-4 rounded-lg border border-white/10 bg-bg-tertiary/60 p-3 text-xs text-gray-300">
          <p className="font-medium mb-1 text-gray-200">team_id の考え方（チーム管理）</p>
          <p>
            `team_id` は「どのチームに利用量・予算を集計するか」の識別子です。個人運用なら空欄、
            研究室/部署単位でまとめるなら同じ `team_id`（例: `lab-a`）を設定します。
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <label className="block">
            <span className="mb-1 block text-sm font-medium text-gray-300">ログインID</span>
            <input
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
              placeholder="例: yamada"
              value={newUser.username}
              onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-sm font-medium text-gray-300">ログインパスワード</span>
            <input
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
              type="password"
              placeholder="password"
              value={newUser.password}
              onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-sm font-medium text-gray-300">権限</span>
            <select
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
              value={newUser.role}
              onChange={(e) => setNewUser({ ...newUser, role: e.target.value as "admin" | "user" })}
            >
              <option value="user">一般ユーザー</option>
              <option value="admin">管理者</option>
            </select>
          </label>
          <label className="block">
            <span className="mb-1 block text-sm font-medium text-gray-300">LiteLLM team_id（任意）</span>
            <input
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
              placeholder="例: lab-a / admins"
              value={newUser.litellm_team_id}
              onChange={(e) => setNewUser({ ...newUser, litellm_team_id: e.target.value })}
            />
          </label>
        </div>
        <button
          onClick={createLocalUser}
          disabled={!newUser.username || !newUser.password}
          className="mt-4 px-4 py-2 bg-accent-primary text-white rounded-lg disabled:opacity-50"
        >
          ログインユーザーを作成
        </button>
        {message && <p className="mt-3 text-sm text-gray-400">{message}</p>}
        <div className="mt-4 space-y-4">
          <div className="rounded-lg border border-white/10">
            <table className="w-full text-sm">
              <thead className="bg-bg-primary text-gray-400">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">ユーザー名</th>
                  <th className="px-3 py-2 text-left font-medium">権限</th>
                  <th className="px-3 py-2 text-left font-medium">LiteLLM user_id</th>
                  <th className="px-3 py-2 text-left font-medium">team_id</th>
                  <th className="px-3 py-2 text-left font-medium">作成日時</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => {
                  const active = selectedUser?.username === user.username;
                  return (
                    <tr
                      key={user.username}
                      className={`border-t border-white/5 bg-bg-tertiary/40 cursor-pointer transition-colors ${
                        active ? "bg-accent-primary/20" : "hover:bg-bg-tertiary/80"
                      }`}
                      onClick={() => handleSelectUser(user)}
                    >
                      <td className="px-3 py-2 font-medium text-white">{user.username}</td>
                      <td className="px-3 py-2 text-gray-300">
                        {user.role === "admin" ? "管理者" : "一般ユーザー"}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs text-gray-300">
                        {user.litellm_user_id || "-"}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs text-gray-300">
                        {user.litellm_team_id || "-"}
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-400">
                        {user.created_at ? new Date(user.created_at * 1000).toLocaleString() : "-"}
                      </td>
                    </tr>
                  );
                })}
                {users.length === 0 && (
                  <tr className="border-t border-white/5 bg-bg-tertiary/20">
                    <td colSpan={5} className="px-3 py-6 text-center text-gray-500">
                      ユーザーがまだありません
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {selectedUser && (
            <div className="rounded-lg border border-white/10 bg-bg-tertiary/40 p-4 space-y-4">
              <h3 className="text-sm font-medium text-gray-200 mb-1">
                選択中ユーザーの詳細 / API キー
              </h3>
              <p className="text-xs text-gray-400 mb-2">
                ロールや team_id、パスワードを更新できます。また、このユーザー用の LiteLLM API キーをワンクリックで発行し、下に一覧表示します。
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-gray-300">ユーザー名</span>
                  <input
                    disabled
                    className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-gray-400"
                    value={selectedUser.username}
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-gray-300">権限</span>
                  <select
                    className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
                    value={editUser.role}
                    onChange={(e) =>
                      setEditUser({ ...editUser, role: e.target.value as "admin" | "user" })
                    }
                  >
                    <option value="user">一般ユーザー</option>
                    <option value="admin">管理者</option>
                  </select>
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-gray-300">
                    LiteLLM user_id
                  </span>
                  <input
                    className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
                    value={editUser.litellm_user_id}
                    onChange={(e) =>
                      setEditUser({ ...editUser, litellm_user_id: e.target.value })
                    }
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-gray-300">
                    team_id（任意）
                  </span>
                  <input
                    className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
                    value={editUser.litellm_team_id}
                    onChange={(e) =>
                      setEditUser({ ...editUser, litellm_team_id: e.target.value })
                    }
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-gray-300">
                    新しいパスワード（空欄なら変更しない）
                  </span>
                  <input
                    type="password"
                    className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
                    value={editUser.newPassword}
                    onChange={(e) =>
                      setEditUser({ ...editUser, newPassword: e.target.value })
                    }
                  />
                </label>
                <label className="flex items-center gap-2 mt-2">
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-white/20 bg-bg-tertiary"
                    checked={editUser.disabled}
                    onChange={(e) =>
                      setEditUser({ ...editUser, disabled: e.target.checked })
                    }
                  />
                  <span className="text-xs text-gray-300">このユーザーを無効化する</span>
                </label>
              </div>
              <button
                onClick={saveUserChanges}
                className="px-3 py-2 text-xs rounded-lg bg-accent-primary text-white"
              >
                ユーザー情報を更新
              </button>

              <div className="mt-4 border-t border-white/10 pt-3 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-gray-300">
                    <Key className="w-3 h-3 text-accent-warning" />
                    <span>このユーザーの LiteLLM API キー</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={refreshLiteKeys}
                      className="inline-flex items-center gap-1 rounded-md border border-white/10 px-2 py-1 text-[11px] text-gray-300 hover:bg-bg-primary"
                    >
                      <RefreshCw className="w-3 h-3" />
                      再読込
                    </button>
                    <button
                      onClick={issueApiKeyForSelectedUser}
                      disabled={!hasModelSelection}
                      className="inline-flex items-center gap-1 rounded-md bg-accent-primary px-2 py-1 text-[11px] text-white disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Key className="w-3 h-3" />
                      このユーザー向けにAPIキー発行
                    </button>
                  </div>
                </div>
                <div className="rounded-lg border border-white/10 bg-bg-tertiary/60 p-3 text-xs text-gray-300 space-y-1">
                  <p>単位の目安: `max_budget` は USD、`RPM` は requests/min、`TPM` は tokens/min です。</p>
                  <p>
                    例: `RPM=10` は「1分に最大10リクエスト」、`TPM=100000` は「1分に最大10万トークン」です。
                  </p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <label className="block">
                    <span className="mb-1 block text-sm font-medium text-gray-300">利用できるモデル</span>
                    <input
                      className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
                      placeholder="例: cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4（複数はカンマ区切り）"
                      value={keyOptions.models}
                      onChange={(e) => setKeyOptions({ ...keyOptions, models: e.target.value })}
                    />
                    <p className="mt-1 text-[11px] text-gray-500">
                      `*` を指定すると全モデルアクセスキーになります。`*` 以外ではモデル名を1つ以上指定してください。
                    </p>
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-sm font-medium text-gray-300">予算上限 (USD)</span>
                    <input
                      className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
                      placeholder="例: 10（空欄で制限なし）"
                      value={keyOptions.max_budget}
                      onChange={(e) => setKeyOptions({ ...keyOptions, max_budget: e.target.value })}
                    />
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-sm font-medium text-gray-300">予算期間</span>
                    <input
                      className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
                      placeholder="例: 30d / 7d / 1mo"
                      value={keyOptions.budget_duration}
                      onChange={(e) => setKeyOptions({ ...keyOptions, budget_duration: e.target.value })}
                    />
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-sm font-medium text-gray-300">RPM 上限 (req/min)</span>
                    <input
                      className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
                      placeholder="例: 60（空欄で制限なし）"
                      value={keyOptions.rpm_limit}
                      onChange={(e) => setKeyOptions({ ...keyOptions, rpm_limit: e.target.value })}
                    />
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-sm font-medium text-gray-300">TPM 上限 (tok/min)</span>
                    <input
                      className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
                      placeholder="例: 100000（空欄で制限なし）"
                      value={keyOptions.tpm_limit}
                      onChange={(e) => setKeyOptions({ ...keyOptions, tpm_limit: e.target.value })}
                    />
                  </label>
                </div>
                {apiMessage && (
                  <p className="text-xs text-gray-400 whitespace-pre-wrap">{apiMessage}</p>
                )}
                <div className="space-y-2">
                  {asKeyEntries(filteredKeys as unknown[]).length > 0 ? (
                    asKeyEntries(filteredKeys as unknown[]).map((entry, idx) => (
                      <div
                        key={entry.key ?? `adm-${idx}`}
                        className="rounded-lg border border-white/10 bg-bg-primary/70 p-3 text-xs"
                      >
                        <ApiKeyCopyBlock entry={entry} index={idx} scope="admin" />
                        <div className="mt-3 flex items-start justify-between gap-3 border-t border-white/10 pt-3">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-gray-300 flex-1">
                            <p><span className="text-gray-500">user_id:</span> {entry.info?.user_id || "-"}</p>
                            <p><span className="text-gray-500">team_id:</span> {entry.info?.team_id || "-"}</p>
                            <p><span className="text-gray-500">models:</span> {(entry.info?.models || []).join(", ") || "-"}</p>
                            <p><span className="text-gray-500">max_budget:</span> {entry.info?.max_budget ?? "-"}</p>
                            <p><span className="text-gray-500">budget_duration:</span> {entry.info?.budget_duration || "-"}</p>
                            <p><span className="text-gray-500">RPM:</span> {entry.info?.rpm_limit ?? "-"}</p>
                            <p><span className="text-gray-500">TPM:</span> {entry.info?.tpm_limit ?? "-"}</p>
                            <p><span className="text-gray-500">作成:</span> {entry.info?.created_at || "-"}</p>
                            <p><span className="text-gray-500">有効期限:</span> {entry.info?.expires || "なし"}</p>
                          </div>
                          <button
                            onClick={() => deleteApiKey(entry.deleteKey || entry.key)}
                            className="shrink-0 rounded-md border border-red-500/40 bg-red-500/10 px-2 py-1 text-[11px] text-red-300 hover:bg-red-500/20"
                          >
                            削除
                          </button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-gray-500 text-xs">
                      まだこのユーザーに紐づく API キーはありません。
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
