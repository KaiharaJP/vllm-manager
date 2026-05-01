"use client";

import { useEffect, useState } from "react";
import { Key, Users } from "lucide-react";
import { api } from "@/lib/api";
import type { AppUser } from "@/types";

export default function LiteLLMAdminPanel() {
  const [users, setUsers] = useState<AppUser[]>([]);
  const [keys, setKeys] = useState<unknown>(null);
  const [teams, setTeams] = useState<unknown>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [newUser, setNewUser] = useState({ username: "", password: "", role: "user" as "admin" | "user", litellm_team_id: "" });
  const [keyForm, setKeyForm] = useState({ user_id: "", team_id: "", models: "vllm-local", max_budget: "", budget_duration: "30d", rpm_limit: "", tpm_limit: "" });

  useEffect(() => {
    refresh();
  }, []);

  async function refresh() {
    const [localUsers, liteKeys, liteTeams] = await Promise.allSettled([
      api.getUsers(),
      api.getLiteLLMKeys(),
      api.getLiteLLMTeams(),
    ]);
    if (localUsers.status === "fulfilled") setUsers(localUsers.value);
    if (liteKeys.status === "fulfilled") setKeys(liteKeys.value);
    if (liteTeams.status === "fulfilled") setTeams(liteTeams.value);
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
      refresh();
    } catch (err) {
      setMessage(String(err));
    }
  }

  async function createKey() {
    setMessage(null);
    try {
      const payload: Record<string, unknown> = {
        user_id: keyForm.user_id || undefined,
        team_id: keyForm.team_id || undefined,
        models: keyForm.models.split(",").map((item) => item.trim()).filter(Boolean),
        budget_duration: keyForm.budget_duration || undefined,
      };
      if (keyForm.max_budget) payload.max_budget = Number(keyForm.max_budget);
      if (keyForm.rpm_limit) payload.rpm_limit = Number(keyForm.rpm_limit);
      if (keyForm.tpm_limit) payload.tpm_limit = Number(keyForm.tpm_limit);
      await api.createLiteLLMKey(payload);
      setMessage("LiteLLM キーを発行しました");
      refresh();
    } catch (err) {
      setMessage(String(err));
    }
  }

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Users className="w-5 h-5 text-accent-primary" />
          vLLM Manager ユーザー
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="username" value={newUser.username} onChange={(e) => setNewUser({ ...newUser, username: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" type="password" placeholder="password" value={newUser.password} onChange={(e) => setNewUser({ ...newUser, password: e.target.value })} />
          <select className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" value={newUser.role} onChange={(e) => setNewUser({ ...newUser, role: e.target.value as "admin" | "user" })}>
            <option value="user">user</option>
            <option value="admin">admin</option>
          </select>
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="LiteLLM team_id" value={newUser.litellm_team_id} onChange={(e) => setNewUser({ ...newUser, litellm_team_id: e.target.value })} />
        </div>
        <button onClick={createLocalUser} disabled={!newUser.username || !newUser.password} className="mt-4 px-4 py-2 bg-accent-primary text-white rounded-lg disabled:opacity-50">
          ユーザー作成
        </button>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-2">
          {users.map((user) => (
            <div key={user.username} className="bg-bg-tertiary rounded-lg p-3 text-sm">
              <p className="font-medium">{user.username} <span className="text-gray-500">({user.role})</span></p>
              <p className="text-xs text-gray-500">LiteLLM: {user.litellm_user_id || "-"} / {user.litellm_team_id || "-"}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Key className="w-5 h-5 text-accent-warning" />
          LiteLLM API キー / 予算 / レート制限
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="user_id" value={keyForm.user_id} onChange={(e) => setKeyForm({ ...keyForm, user_id: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="team_id" value={keyForm.team_id} onChange={(e) => setKeyForm({ ...keyForm, team_id: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="models CSV" value={keyForm.models} onChange={(e) => setKeyForm({ ...keyForm, models: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="max_budget USD" value={keyForm.max_budget} onChange={(e) => setKeyForm({ ...keyForm, max_budget: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="budget_duration" value={keyForm.budget_duration} onChange={(e) => setKeyForm({ ...keyForm, budget_duration: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="rpm_limit" value={keyForm.rpm_limit} onChange={(e) => setKeyForm({ ...keyForm, rpm_limit: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="tpm_limit" value={keyForm.tpm_limit} onChange={(e) => setKeyForm({ ...keyForm, tpm_limit: e.target.value })} />
        </div>
        <button onClick={createKey} className="mt-4 px-4 py-2 bg-accent-primary text-white rounded-lg">
          キー発行
        </button>
        {message && <p className="mt-3 text-sm text-gray-400">{message}</p>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <JsonPanel title="LiteLLM Keys" value={keys} />
        <JsonPanel title="LiteLLM Teams" value={teams} />
      </div>
    </div>
  );
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
      <h3 className="text-sm font-medium text-gray-400 mb-3">{title}</h3>
      <pre className="bg-bg-primary rounded-lg p-3 text-xs overflow-auto max-h-96">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}
