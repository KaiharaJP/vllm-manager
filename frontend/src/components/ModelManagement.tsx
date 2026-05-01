"use client";

import { useState } from "react";
import { Download, Plus } from "lucide-react";
import { api } from "@/lib/api";
import type { DownloadJob, Model } from "@/types";

interface ModelManagementProps {
  models: Model[];
  jobs: DownloadJob[];
  onChanged: () => void;
}

export default function ModelManagement({ models, jobs, onChanged }: ModelManagementProps) {
  const [form, setForm] = useState({
    id: "",
    name: "",
    size: "",
    revision: "",
    recommended_context_length: 8192,
    gated: false,
    trust_remote_code: false,
  });
  const [message, setMessage] = useState<string | null>(null);

  async function registerModel() {
    setMessage(null);
    try {
      await api.registerModel({
        ...form,
        revision: form.revision || null,
        name: form.name || form.id,
        size: form.size || "unknown",
      });
      setForm({ ...form, id: "", name: "", size: "", revision: "" });
      setMessage("モデルを登録しました");
      onChanged();
    } catch (err) {
      setMessage(String(err));
    }
  }

  async function startDownload(modelId: string) {
    setMessage(null);
    try {
      await api.startModelDownload(modelId);
      setMessage("ダウンロードジョブを開始しました");
      onChanged();
    } catch (err) {
      setMessage(String(err));
    }
  }

  function formatBytes(value = 0) {
    if (!value) return "-";
    const gb = value / 1024 / 1024 / 1024;
    return `${gb.toFixed(2)} GB`;
  }

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Plus className="w-5 h-5 text-accent-primary" />
          管理者モデル登録
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="repo_id" value={form.id} onChange={(e) => setForm({ ...form, id: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="表示名" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="サイズ 例: 8B" value={form.size} onChange={(e) => setForm({ ...form, size: e.target.value })} />
          <input className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="revision 任意" value={form.revision} onChange={(e) => setForm({ ...form, revision: e.target.value })} />
        </div>
        <div className="flex gap-4 mt-4 text-sm text-gray-400">
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={form.gated} onChange={(e) => setForm({ ...form, gated: e.target.checked })} />
            gated model
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={form.trust_remote_code} onChange={(e) => setForm({ ...form, trust_remote_code: e.target.checked })} />
            trust remote code
          </label>
        </div>
        <button
          onClick={registerModel}
          disabled={!form.id}
          className="mt-4 px-4 py-2 bg-accent-primary text-white rounded-lg disabled:opacity-50"
        >
          登録
        </button>
        {message && <p className="mt-3 text-sm text-gray-400">{message}</p>}
      </div>

      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Download className="w-5 h-5 text-accent-success" />
          モデル一覧 / ダウンロード
        </h2>
        <div className="space-y-3">
          {models.map((model) => {
            const activeJob = jobs.find((job) => job.model_id === model.id && ["queued", "running"].includes(job.status));
            return (
              <div key={model.id} className="bg-bg-tertiary rounded-lg p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                <div>
                  <p className="font-medium">{model.name}</p>
                  <p className="text-xs text-gray-500 font-mono">{model.id}</p>
                  <p className="text-xs text-gray-500">
                    {model.size} / {model.downloaded ? `downloaded ${formatBytes(model.cache_size_bytes)}` : "not downloaded"}
                  </p>
                  {activeJob && (
                    <div className="mt-2">
                      <div className="h-2 bg-bg-primary rounded-full overflow-hidden">
                        <div className="h-full bg-accent-primary" style={{ width: `${activeJob.progress}%` }} />
                      </div>
                      <p className="text-xs text-gray-500 mt-1">{activeJob.status}: {activeJob.progress}%</p>
                    </div>
                  )}
                </div>
                <button
                  onClick={() => startDownload(model.id)}
                  disabled={Boolean(activeJob)}
                  className="px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg disabled:opacity-50"
                >
                  ダウンロード
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
