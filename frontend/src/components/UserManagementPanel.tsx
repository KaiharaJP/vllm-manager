"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { AppUser, ApiToken, Model } from "@/types";
import { ClipboardCopy, Info, Key, KeyRound, RefreshCw, Trash2, Users } from "lucide-react";
import { api } from "@/lib/api";

interface UserManagementPanelProps {
  currentUser: AppUser;
}

const MODEL_SELECTOR_HINT =
  "API キーが使える model 名を指定します。クライアント（curl / OpenAI SDK / Claude Code 等）がリクエストに書く名前と完全一致する必要があります。LiteLLM 経由（:14000）なら短いエイリアス、backend 直（:18000）なら Hugging Face ID を選ぶことが多いです。";

const ALL_MODELS_HINT =
  "どんな model 名でも利用可能なキーになります（制限なし）。個人・検証向けで、本番や共有キーでは漏洩時の影響が大きいため慎重に。選択すると下の個別指定は無効になります。";

const ALIAS_SECTION_HINT =
  "LiteLLM の設定（litellm_config.yaml）で決めた固定名です。実モデル ID の代わりに短い名前で叩けます。backend が「いま動いている vLLM」へ自動でつなぎます。";

const CATALOG_SECTION_HINT =
  "vLLM Manager に登録されている実モデル ID です。クライアントが model: \"org/name\" のようにフル ID で指定する場合に選びます。特定モデルだけ使わせたいユーザー向けの制限に向いています。";

const CUSTOM_MODELS_HINT =
  "上記一覧にない model 名を追加できます。LiteLLM またはクライアントが実際に送る文字列と完全一致させてください。複数指定する場合はカンマ区切りです。";

function InfoTooltip({ text }: { text: ReactNode }) {
  return (
    <span className="group relative inline-flex shrink-0">
      <Info className="h-4 w-4 cursor-help text-gray-500 transition-colors group-hover:text-accent-primary" />
      <span className="pointer-events-none absolute left-1/2 top-6 z-30 hidden w-72 -translate-x-1/2 rounded-lg border border-white/10 bg-bg-primary p-3 text-xs font-normal leading-relaxed text-gray-300 shadow-xl group-hover:block">
        {text}
      </span>
    </span>
  );
}

function HintLabel({
  label,
  hint,
  className = "text-sm font-medium text-gray-300",
}: {
  label: ReactNode;
  hint: ReactNode;
  className?: string;
}) {
  return (
    <span className={`inline-flex items-center gap-1 ${className}`}>
      {label}
      <InfoTooltip text={hint} />
    </span>
  );
}

const LITELLM_ROUTE_ALIASES: {
  id: string;
  label: string;
  detail: string;
}[] = [
  {
    id: "vllm-local",
    label: "汎用エイリアス（迷ったらこれ）",
    detail:
      "LiteLLM（:14000）経由で model に \"vllm-local\" と書くだけで、いま起動中の vLLM に接続します。Hugging Face の長いモデル ID をクライアント側に覚えなくてよく、サーバーでモデルを差し替えても設定変更が不要です。",
  },
  {
    id: "claude-vllm-local",
    label: "vllm-local と同じ転送先（別名）",
    detail:
      "中身は vllm-local と同じ backend → vLLM です。Claude Code などで model 名を分けて管理したいとき向け。Claude Code 専用という意味ではなく、キーとクライアントの model 名をこの名前に揃えれば使えます。",
  },
];

function ApiKeyModelSelector({
  catalogModels,
  selectedModels,
  allModelsAccess,
  customModels,
  onToggleAllModelsAccess,
  onToggleModel,
  onCustomModelsChange,
}: {
  catalogModels: Model[];
  selectedModels: string[];
  allModelsAccess: boolean;
  customModels: string;
  onToggleAllModelsAccess: (checked: boolean) => void;
  onToggleModel: (modelId: string, checked: boolean) => void;
  onCustomModelsChange: (value: string) => void;
}) {
  const aliasIds = new Set(LITELLM_ROUTE_ALIASES.map((a) => a.id));
  const catalogOnly = catalogModels.filter((m) => !aliasIds.has(m.id));

  return (
    <div className="md:col-span-3 space-y-3">
      <HintLabel label="利用できるモデル" hint={MODEL_SELECTOR_HINT} />
      <div className="rounded-lg border border-white/10 bg-bg-tertiary/60 p-3 space-y-3">
        <label className="flex items-start gap-2 cursor-pointer">
          <input
            type="checkbox"
            className="mt-0.5 h-4 w-4 rounded border-white/20 bg-bg-tertiary"
            checked={allModelsAccess}
            onChange={(e) => onToggleAllModelsAccess(e.target.checked)}
          />
          <HintLabel
            className="text-xs font-medium text-gray-200"
            label="全モデル（`*`）"
            hint={ALL_MODELS_HINT}
          />
        </label>

        {!allModelsAccess && (
          <>
            <div className="border-t border-white/10 pt-3 space-y-2">
              <HintLabel
                className="text-[11px] font-medium text-gray-400"
                label="LiteLLM ルートエイリアス"
                hint={ALIAS_SECTION_HINT}
              />
              <div className="space-y-1.5">
                {LITELLM_ROUTE_ALIASES.map((alias) => (
                  <label key={alias.id} className="flex items-start gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="mt-0.5 h-4 w-4 shrink-0 rounded border-white/20 bg-bg-tertiary"
                      checked={selectedModels.includes(alias.id)}
                      onChange={(e) => onToggleModel(alias.id, e.target.checked)}
                    />
                    <span className="inline-flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-xs text-gray-300 min-w-0">
                      <code className="text-gray-100">{alias.id}</code>
                      <span className="text-gray-400">{alias.label}</span>
                      <InfoTooltip text={alias.detail} />
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <div className="border-t border-white/10 pt-3 space-y-2">
              <HintLabel
                className="text-[11px] font-medium text-gray-400"
                label="登録済みモデル（Hugging Face ID）"
                hint={CATALOG_SECTION_HINT}
              />
              {catalogOnly.length > 0 ? (
                <div className="max-h-40 overflow-y-auto space-y-1.5 pr-1">
                  {catalogOnly.map((model) => (
                    <label key={model.id} className="flex items-start gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        className="mt-0.5 h-4 w-4 shrink-0 rounded border-white/20 bg-bg-tertiary"
                        checked={selectedModels.includes(model.id)}
                        onChange={(e) => onToggleModel(model.id, e.target.checked)}
                      />
                      <span className="text-xs text-gray-300 break-all min-w-0">
                        <code className="text-gray-100">{model.id}</code>
                        {model.size ? (
                          <span className="text-gray-500 ml-1.5">({model.size})</span>
                        ) : null}
                        {model.downloaded ? (
                          <span className="text-emerald-400/80 ml-1.5">DL済</span>
                        ) : (
                          <span className="text-amber-400/80 ml-1.5">未DL</span>
                        )}
                      </span>
                    </label>
                  ))}
                </div>
              ) : (
                <p className="text-[11px] text-gray-500">
                  モデルカタログが空です。「モデル管理」でモデルを登録してください。
                </p>
              )}
            </div>

            <div className="border-t border-white/10 pt-3 space-y-1.5">
              <label className="block">
                <span className="mb-1 inline-flex items-center gap-1 text-[11px] font-medium text-gray-400">
                  その他（カンマ区切りで手入力）
                  <InfoTooltip text={CUSTOM_MODELS_HINT} />
                </span>
                <input
                  className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
                  placeholder="例: custom/model-name"
                  value={customModels}
                  onChange={(e) => onCustomModelsChange(e.target.value)}
                />
              </label>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function ApiTokenPanel({
  tokens,
  onRefresh,
  onRevoke,
  onCreate,
  createBusy,
  justCreatedToken,
  message,
}: {
  tokens: ApiToken[];
  onRefresh: () => void;
  onRevoke: (tokenId: string) => void;
  onCreate?: (name: string, expiresInDays: number | undefined) => void;
  createBusy?: boolean;
  justCreatedToken: string | null;
  message: string | null;
}) {
  const [name, setName] = useState("");
  const [expiresInDays, setExpiresInDays] = useState("");

  return (
    <div className="rounded-lg border border-white/10 bg-bg-tertiary/40 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-gray-300">
          <KeyRound className="w-3 h-3 text-accent-warning" />
          <span>永続APIトークン（PAT）― サーバー操作・自動化用</span>
        </div>
        <button
          onClick={onRefresh}
          className="inline-flex items-center gap-1 rounded-md border border-white/10 px-2 py-1 text-[11px] text-gray-300 hover:bg-bg-primary"
        >
          <RefreshCw className="w-3 h-3" />
          再読込
        </button>
      </div>
      <p className="text-[11px] text-gray-500">
        LiteLLM 推論用キー（sk-...）とは別物です。<code>vllm-cli.sh</code>
        やスクリプトから、ログインなしでサーバー起動/停止・モデルダウンロードを自動化するためのトークンです。
      </p>
      {onCreate && (
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
          <label className="flex-1 block">
            <span className="mb-1 block text-[11px] font-medium text-gray-400">名前</span>
            <input
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
              placeholder="例: cron-job"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label className="w-32 block">
            <span className="mb-1 block text-[11px] font-medium text-gray-400">有効日数（任意）</span>
            <input
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-xs"
              placeholder="無期限"
              value={expiresInDays}
              onChange={(e) => setExpiresInDays(e.target.value)}
            />
          </label>
          <button
            disabled={!name.trim() || createBusy}
            onClick={() => {
              onCreate(name.trim(), expiresInDays ? Number(expiresInDays) : undefined);
              setName("");
              setExpiresInDays("");
            }}
            className="shrink-0 inline-flex items-center gap-1 rounded-md bg-accent-primary px-3 py-2 text-[11px] text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            <KeyRound className="w-3 h-3" />
            発行
          </button>
        </div>
      )}
      {justCreatedToken && (
        <div className="rounded-lg border border-accent-warning/40 bg-accent-warning/10 p-3 text-xs text-accent-warning space-y-1">
          <p className="font-medium">この画面でしか表示されません。今すぐコピーしてください。</p>
          <code className="block break-all text-[11px] text-gray-100">{justCreatedToken}</code>
        </div>
      )}
      {message && <p className="text-[11px] text-gray-400 whitespace-pre-wrap">{message}</p>}
      <div className="space-y-2">
        {tokens.length > 0 ? (
          tokens.map((token) => (
            <div key={token.id} className="rounded-lg border border-white/10 bg-bg-primary/70 p-3 text-xs">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="font-medium text-gray-100">
                    {token.name}
                    {token.disabled && <span className="ml-2 text-[10px] text-accent-danger">失効済み</span>}
                  </p>
                  <p className="text-gray-500 font-mono">{token.prefix}...</p>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-gray-400">
                    <span>作成: {new Date(token.created_at * 1000).toLocaleString()}</span>
                    <span>
                      最終使用:{" "}
                      {token.last_used_at ? new Date(token.last_used_at * 1000).toLocaleString() : "未使用"}
                    </span>
                    <span>
                      有効期限:{" "}
                      {token.expires_at ? new Date(token.expires_at * 1000).toLocaleString() : "無期限"}
                    </span>
                  </div>
                </div>
                {!token.disabled && (
                  <button
                    onClick={() => onRevoke(token.id)}
                    className="shrink-0 inline-flex items-center gap-1 rounded-md border border-red-500/40 bg-red-500/10 px-2 py-1 text-[11px] text-red-300 hover:bg-red-500/20"
                  >
                    <Trash2 className="w-3 h-3" />
                    失効
                  </button>
                )}
              </div>
            </div>
          ))
        ) : (
          <p className="text-gray-500 text-xs">まだ発行済みのトークンはありません。</p>
        )}
      </div>
    </div>
  );
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
  const [catalogModels, setCatalogModels] = useState<Model[]>([]);
  const [keyOptions, setKeyOptions] = useState({
    selectedModels: ["vllm-local"],
    allModelsAccess: false,
    customModels: "",
    max_budget: "",
    budget_duration: "",
    rpm_limit: "",
    tpm_limit: "",
  });
  const [copiedKeySig, setCopiedKeySig] = useState<string | null>(null);
  const [issuedSecretsByToken, setIssuedSecretsByToken] = useState<Record<string, string>>({});
  const [myTokens, setMyTokens] = useState<ApiToken[]>([]);
  const [myTokenMessage, setMyTokenMessage] = useState<string | null>(null);
  const [myTokenCreating, setMyTokenCreating] = useState(false);
  const [justCreatedMyToken, setJustCreatedMyToken] = useState<string | null>(null);
  const [selectedUserTokens, setSelectedUserTokens] = useState<ApiToken[]>([]);
  const [selectedUserTokenMessage, setSelectedUserTokenMessage] = useState<string | null>(null);

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
      api.getModels().then(setCatalogModels).catch(() => setCatalogModels([]));
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
    refreshMyTokens();
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

  async function refreshMyTokens() {
    try {
      const result = await api.getMyTokens();
      setMyTokens(result.tokens);
    } catch (err) {
      setMyTokenMessage(String(err));
    }
  }

  async function createMyToken(name: string, expiresInDays: number | undefined) {
    setMyTokenCreating(true);
    setMyTokenMessage(null);
    try {
      const created = await api.createMyToken({ name, expires_in_days: expiresInDays });
      setJustCreatedMyToken(created.token);
      await refreshMyTokens();
    } catch (err) {
      setMyTokenMessage(String(err));
    } finally {
      setMyTokenCreating(false);
    }
  }

  async function revokeMyToken(tokenId: string) {
    setMyTokenMessage(null);
    try {
      await api.revokeMyToken(tokenId);
      setMyTokenMessage("トークンを失効させました。");
      await refreshMyTokens();
    } catch (err) {
      setMyTokenMessage(String(err));
    }
  }

  async function refreshSelectedUserTokens() {
    if (!selectedUser) return;
    try {
      const result = await api.getUserTokens(selectedUser.username);
      setSelectedUserTokens(result.tokens);
    } catch (err) {
      setSelectedUserTokenMessage(String(err));
    }
  }

  async function revokeSelectedUserToken(tokenId: string) {
    if (!selectedUser) return;
    setSelectedUserTokenMessage(null);
    try {
      await api.revokeUserToken(selectedUser.username, tokenId);
      setSelectedUserTokenMessage("トークンを強制失効させました。");
      await refreshSelectedUserTokens();
    } catch (err) {
      setSelectedUserTokenMessage(String(err));
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
      refreshSelectedUserTokens();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedUser?.username]);

  const filteredKeys = useMemo(() => {
    if (!liteKeys) return [];
    return Array.isArray(liteKeys) ? liteKeys : [];
  }, [liteKeys, selectedUser]);

  const resolvedModels = useMemo(() => {
    if (keyOptions.allModelsAccess) return ["*"];
    const custom = parseModels(keyOptions.customModels);
    const seen = new Set<string>();
    const result: string[] = [];
    for (const modelId of [...keyOptions.selectedModels, ...custom]) {
      if (modelId && !seen.has(modelId)) {
        seen.add(modelId);
        result.push(modelId);
      }
    }
    return result;
  }, [keyOptions.allModelsAccess, keyOptions.selectedModels, keyOptions.customModels]);
  const hasModelSelection = resolvedModels.length > 0;

  function toggleAllModelsAccess(checked: boolean) {
    setKeyOptions((prev) => ({
      ...prev,
      allModelsAccess: checked,
      selectedModels:
        checked || prev.selectedModels.length > 0 ? prev.selectedModels : ["vllm-local"],
    }));
  }

  function toggleSelectedModel(modelId: string, checked: boolean) {
    setKeyOptions((prev) => ({
      ...prev,
      allModelsAccess: false,
      selectedModels: checked
        ? [...prev.selectedModels, modelId]
        : prev.selectedModels.filter((id) => id !== modelId),
    }));
  }

  function buildKeyPayload() {
    const payload: {
      models: string[];
      max_budget?: number;
      budget_duration?: string;
      rpm_limit?: number;
      tpm_limit?: number;
    } = { models: resolvedModels };
    if (keyOptions.max_budget) payload.max_budget = Number(keyOptions.max_budget);
    if (keyOptions.budget_duration) payload.budget_duration = keyOptions.budget_duration;
    if (keyOptions.rpm_limit) payload.rpm_limit = Number(keyOptions.rpm_limit);
    if (keyOptions.tpm_limit) payload.tpm_limit = Number(keyOptions.tpm_limit);
    return payload;
  }

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
        if (!resolvedModels.length) {
          setApiMessage("モデル名を1つ以上指定してください。`*` を指定すると全モデルアクセスになります。");
          return;
        }
        const created = await api.createMyApiKey(buildKeyPayload());
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
      if (!resolvedModels.length) {
        setApiMessage("モデル名を1つ以上指定してください。`*` を指定すると全モデルアクセスになります。");
        return;
      }
      const created = await api.createUserApiKey(selectedUser.username, buildKeyPayload());
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

        <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
          <ApiTokenPanel
            tokens={myTokens}
            onRefresh={refreshMyTokens}
            onRevoke={revokeMyToken}
            onCreate={createMyToken}
            createBusy={myTokenCreating}
            justCreatedToken={justCreatedMyToken}
            message={myTokenMessage}
          />
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
                  <ApiKeyModelSelector
                    catalogModels={catalogModels}
                    selectedModels={keyOptions.selectedModels}
                    allModelsAccess={keyOptions.allModelsAccess}
                    customModels={keyOptions.customModels}
                    onToggleAllModelsAccess={toggleAllModelsAccess}
                    onToggleModel={toggleSelectedModel}
                    onCustomModelsChange={(value) =>
                      setKeyOptions((prev) => ({ ...prev, customModels: value, allModelsAccess: false }))
                    }
                  />
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

              <div className="mt-4 border-t border-white/10 pt-3">
                <ApiTokenPanel
                  tokens={selectedUserTokens}
                  onRefresh={refreshSelectedUserTokens}
                  onRevoke={revokeSelectedUserToken}
                  justCreatedToken={null}
                  message={selectedUserTokenMessage}
                />
                <p className="mt-2 text-[11px] text-gray-500">
                  管理者はこのユーザーの永続APIトークンを閲覧・強制失効できますが、代理発行はできません（発行はユーザー本人がマイページから行います）。
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
