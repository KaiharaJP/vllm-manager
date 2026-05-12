"use client";

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Info, Key } from "lucide-react";
import { api } from "@/lib/api";

export default function LiteLLMAdminPanel() {
  const [keys, setKeys] = useState<unknown>(null);
  const [teams, setTeams] = useState<unknown>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [keyForm, setKeyForm] = useState({ user_id: "", team_id: "", models: "vllm-local", max_budget: "", budget_duration: "30d", rpm_limit: "", tpm_limit: "" });

  useEffect(() => {
    refresh();
  }, []);

  async function refresh() {
    const [liteKeys, liteTeams] = await Promise.allSettled([
      api.getLiteLLMKeys(),
      api.getLiteLLMTeams(),
    ]);
    if (liteKeys.status === "fulfilled") setKeys(liteKeys.value);
    if (liteTeams.status === "fulfilled") setTeams(liteTeams.value);
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
        <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
          <Key className="w-5 h-5 text-accent-warning" />
          APIキー / チーム管理
          <InfoTooltip text="外部アプリや curl から LiteLLM Proxy を呼ぶための Bearer token を発行します。予算や RPM/TPM を空欄にすると制限なしとして扱われます。" />
        </h2>
        <div className="mb-4 rounded-lg border border-white/10 bg-bg-tertiary/60 p-3 text-xs text-gray-300 space-y-1">
          <p>単位の目安: `max_budget` は USD、`RPM` は requests/min、`TPM` は tokens/min です。</p>
          <p>
            例: `RPM=10` は「1分に最大10リクエスト」、`TPM=100000` は「1分に最大10万トークン」です。
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Field label="対象ユーザーID" hint="上で作成したログインIDを入れると紐づけやすいです">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="例: yamada" value={keyForm.user_id} onChange={(e) => setKeyForm({ ...keyForm, user_id: e.target.value })} />
          </Field>
          <Field label="対象 team_id" hint="チーム予算・チーム利用量として管理する場合だけ指定します">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="例: lab-a" value={keyForm.team_id} onChange={(e) => setKeyForm({ ...keyForm, team_id: e.target.value })} />
          </Field>
          <Field label="利用できるモデル" hint="通常は vllm-local のままでOK。複数はカンマ区切りです">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="vllm-local" value={keyForm.models} onChange={(e) => setKeyForm({ ...keyForm, models: e.target.value })} />
          </Field>
          <Field label="予算上限 (USD)" hint="通貨は米ドル。例: 10。空欄なら予算制限なし">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="例: 10 (USD)" value={keyForm.max_budget} onChange={(e) => setKeyForm({ ...keyForm, max_budget: e.target.value })} />
          </Field>
          <Field label="予算期間" hint="例: 30d / 7d / 1mo。空欄なら期間指定なし">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="30d" value={keyForm.budget_duration} onChange={(e) => setKeyForm({ ...keyForm, budget_duration: e.target.value })} />
          </Field>
          <Field label="RPM 上限 (requests/min)" hint="1分あたりのリクエスト数。空欄なら制限なし">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="例: 60 req/min" value={keyForm.rpm_limit} onChange={(e) => setKeyForm({ ...keyForm, rpm_limit: e.target.value })} />
          </Field>
          <Field label="TPM 上限 (tokens/min)" hint="1分あたりのトークン数。空欄なら制限なし">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="例: 100000 tok/min" value={keyForm.tpm_limit} onChange={(e) => setKeyForm({ ...keyForm, tpm_limit: e.target.value })} />
          </Field>
        </div>
        <button onClick={createKey} className="mt-4 px-4 py-2 bg-accent-primary text-white rounded-lg">
          推論APIキーを発行
        </button>
        {message && <p className="mt-3 text-sm text-gray-400">{message}</p>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <JsonPanel title="発行済み API キー情報" value={keys} />
        <JsonPanel title="LiteLLM チーム情報" value={teams} />
      </div>
    </div>
  );
}

function InfoTooltip({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex">
      <Info className="h-4 w-4 cursor-help text-gray-500 transition-colors group-hover:text-accent-primary" />
      <span className="pointer-events-none absolute left-1/2 top-6 z-30 hidden w-72 -translate-x-1/2 rounded-lg border border-white/10 bg-bg-primary p-3 text-xs font-normal leading-relaxed text-gray-300 shadow-xl group-hover:block">
        {text}
      </span>
    </span>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 flex items-center gap-1 text-sm font-medium text-gray-300">
        {label}
        <InfoTooltip text={hint} />
      </span>
      {children}
    </label>
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
