"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Download, EyeOff, Info, Plus } from "lucide-react";
import { api, ApiRequestError } from "@/lib/api";
import type { DownloadJob, Model } from "@/types";

type TaskType = "chat" | "embedding" | "rerank";
type ModelTabKey = "llm" | "embedding" | "reranker";

const MODEL_TABS: { key: ModelTabKey; label: string; taskType: TaskType }[] = [
  { key: "llm", label: "LLM", taskType: "chat" },
  { key: "embedding", label: "Embedding", taskType: "embedding" },
  { key: "reranker", label: "Reranker", taskType: "rerank" },
];

function taskTypeLabel(taskType?: string | null): string {
  if (taskType === "embedding") return "embedding";
  if (taskType === "rerank") return "rerank";
  return "LLM";
}

function modelMatchesTab(model: Model, tab: ModelTabKey): boolean {
  const task = model.task_type || "chat";
  if (tab === "llm") return task === "chat";
  if (tab === "embedding") return task === "embedding";
  return task === "rerank";
}

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
    output_dimension: "",
    license_note: "",
    gated: false,
    trust_remote_code: false,
    task_type: "chat" as TaskType,
  });
  const [message, setMessage] = useState<string | null>(null);
  const [resumingModelId, setResumingModelId] = useState<string | null>(null);
  const [modelTab, setModelTab] = useState<ModelTabKey>("llm");
  const [hiddenModelIds, setHiddenModelIds] = useState<string[]>([]);
  const [jobSpeedBytesPerSec, setJobSpeedBytesPerSec] = useState<Record<string, number>>({});
  const [jobSnapshot, setJobSnapshot] = useState<Record<string, { bytes: number; ts: number }>>({});

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const saved = window.localStorage.getItem("vllm_hidden_models");
      if (!saved) return;
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed)) {
        setHiddenModelIds(parsed.filter((v): v is string => typeof v === "string"));
      }
    } catch {
      // ignore malformed localStorage
    }
  }, []);

  useEffect(() => {
    const now = Date.now() / 1000;
    setJobSnapshot((prev) => {
      const next = { ...prev };
      const speed: Record<string, number> = {};
      for (const job of jobs) {
        if (!["queued", "running"].includes(job.status)) continue;
        const prevPoint = prev[job.id];
        const currentBytes = job.downloaded_bytes || 0;
        const currentTs = job.updated_at || now;
        if (prevPoint) {
          const deltaBytes = currentBytes - prevPoint.bytes;
          const deltaSec = Math.max(currentTs - prevPoint.ts, 1e-3);
          speed[job.id] = deltaBytes > 0 ? deltaBytes / deltaSec : 0;
        }
        next[job.id] = { bytes: currentBytes, ts: currentTs };
      }
      setJobSpeedBytesPerSec((prevSpeed) => ({ ...prevSpeed, ...speed }));
      return next;
    });
  }, [jobs]);

  function saveHidden(next: string[]) {
    setHiddenModelIds(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("vllm_hidden_models", JSON.stringify(next));
    }
  }

  function removeFromVisibleList(modelId: string) {
    if (hiddenModelIds.includes(modelId)) return;
    const next = [...hiddenModelIds, modelId];
    saveHidden(next);
    setMessage("一覧表示から削除しました（モデルデータは削除していません）");
  }

  function resetHiddenModels() {
    saveHidden([]);
    setMessage("非表示モデルを再表示しました");
  }

  async function registerModel() {
    setMessage(null);
    try {
      const outputDimension = form.output_dimension.trim()
        ? Number.parseInt(form.output_dimension, 10)
        : undefined;
      await api.registerModel({
        ...form,
        revision: form.revision || null,
        name: form.name || form.id,
        size: form.size || "unknown",
        output_dimension: Number.isFinite(outputDimension) ? outputDimension : undefined,
        license_note: form.license_note.trim() || undefined,
      });
      setForm({
        id: "",
        name: "",
        size: "",
        revision: "",
        recommended_context_length: 8192,
        output_dimension: "",
        license_note: "",
        gated: false,
        trust_remote_code: false,
        task_type: form.task_type,
      });
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

  async function cancelAndResume(modelId: string) {
    if (resumingModelId) return;
    setMessage(null);
    setResumingModelId(modelId);
    try {
      const job = await api.resumeModelDownload(modelId);
      const resumed = jobUsesByteProgress(job)
        ? `${formatBytes(job.downloaded_bytes)} / ${formatBytes(job.total_bytes)} (${jobProgressPercent(job)}%)`
        : `${job.progress}%`;
      setMessage(`続きから再開しました: ${resumed}`);
      onChanged();
    } catch (err) {
      if (err instanceof ApiRequestError && err.status === 401) {
        setMessage("セッションが失効しました。再ログインしてください。");
      } else {
        setMessage(err instanceof ApiRequestError ? err.message : String(err));
      }
    } finally {
      setResumingModelId(null);
    }
  }

  async function updateTaskType(model: Model, task_type: TaskType) {
    if ((model.task_type || "chat") === task_type) return;
    setMessage(null);
    try {
      await api.registerModel({
        id: model.id,
        name: model.name,
        size: model.size,
        revision: model.revision || undefined,
        gated: Boolean(model.gated),
        trust_remote_code: Boolean(model.trust_remote_code),
        recommended_context_length: model.recommended_context_length || 8192,
        output_dimension: model.output_dimension || undefined,
        license_note: model.license_note || undefined,
        task_type,
        source: model.source === "default" ? "custom" : model.source,
      });
      setMessage(`${model.name} の用途を ${taskTypeLabel(task_type)} に変更しました`);
      if (task_type === "chat") setModelTab("llm");
      else if (task_type === "embedding") setModelTab("embedding");
      else setModelTab("reranker");
      onChanged();
    } catch (err) {
      setMessage(err instanceof ApiRequestError ? err.message : String(err));
    }
  }

  async function removeModel(modelId: string) {
    if (!window.confirm(`モデルを完全削除しますか？\n${modelId}\n\n一覧から削除し、ダウンロード済みキャッシュも削除します。`)) return;
    setMessage(null);
    try {
      const result = await api.deleteModel(modelId);
      const freedGb = (result.bytes_freed / 1024 / 1024 / 1024).toFixed(2);
      if (result.removed || result.cache_deleted) {
        setMessage(`モデルを完全削除しました（約 ${freedGb} GB 解放）`);
      } else {
        setMessage("削除対象が見つかりませんでした");
      }
      onChanged();
    } catch (err) {
      setMessage(String(err));
    }
  }

  function formatBytes(value = 0) {
    if (!value) return "-";
    const gb = value / 1024 / 1024 / 1024;
    if (gb >= 1) return `${gb.toFixed(2)} GB`;
    const mb = value / 1024 / 1024;
    if (mb >= 1) return `${mb.toFixed(1)} MB`;
    const kb = value / 1024;
    if (kb >= 1) return `${kb.toFixed(0)} KB`;
    return `${value} B`;
  }

  function jobUsesByteProgress(job: DownloadJob) {
    return job.total_bytes >= 1_000_000;
  }

  function jobProgressPercent(job: DownloadJob) {
    if (jobUsesByteProgress(job) && job.downloaded_bytes > 0) {
      return Math.min(100, Math.round((job.downloaded_bytes / job.total_bytes) * 100));
    }
    return job.progress;
  }

  function jobProgressLabel(job: DownloadJob) {
    if (jobUsesByteProgress(job)) {
      return `${formatBytes(job.downloaded_bytes)} / ${formatBytes(job.total_bytes)} (${jobProgressPercent(job)}%)`;
    }
    if (job.downloaded_bytes > 0) {
      return `${formatBytes(job.downloaded_bytes)} (${job.progress}%)`;
    }
    return `${job.progress}%`;
  }

  const formatSpeed = useMemo(
    () => (value = 0) => {
      if (!value || value <= 0) return "0 B/s";
      const mb = value / 1024 / 1024;
      if (mb >= 1) return `${mb.toFixed(2)} MB/s`;
      const kb = value / 1024;
      if (kb >= 1) return `${kb.toFixed(1)} KB/s`;
      return `${Math.round(value)} B/s`;
    },
    []
  );

  function formatElapsed(seconds: number) {
    if (seconds < 60) return `${Math.max(1, Math.floor(seconds))}秒`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}分`;
    return `${Math.floor(seconds / 3600)}時間`;
  }

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
          <Plus className="w-5 h-5 text-accent-primary" />
          1. Hugging Face モデルを登録
          <InfoTooltip text="ここでは、このアプリで扱うモデル候補だけを登録します。実ファイルのダウンロードは下の一覧から実行します。" />
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <Field label="Hugging Face repo_id" hint="LLM 例: Qwen/Qwen2.5-7B-Instruct / embedding 例: jinaai/jina-embeddings-v3 / rerank 例: BAAI/bge-reranker-v2-m3">
            <input
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
              placeholder={
                form.task_type === "embedding"
                  ? "organization/embedding-model"
                  : form.task_type === "rerank"
                    ? "organization/reranker-model"
                    : "organization/model-name"
              }
              value={form.id}
              onChange={(e) => setForm({ ...form, id: e.target.value })}
            />
          </Field>
          <Field label="画面に表示する名前" hint="空欄なら repo_id をそのまま使います">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="例: Qwen3.5 9B FP8" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </Field>
          <Field label="モデルサイズの目安" hint="一覧で見分けるための表示用です">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="例: 7B / 9B / 32B" value={form.size} onChange={(e) => setForm({ ...form, size: e.target.value })} />
          </Field>
          <Field label="revision / branch" hint="通常は空欄でOK。特定の branch・commit を使う場合だけ指定します">
            <input className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2" placeholder="main / commit hash など" value={form.revision} onChange={(e) => setForm({ ...form, revision: e.target.value })} />
          </Field>
          <Field label="用途 (task_type)" hint="LLM = 生成・チャット、embedding = ベクトル埋め込み、rerank = スコアリング（pooling 系は --runner pooling）">
            <select
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
              value={form.task_type}
              onChange={(e) => {
                const task_type = e.target.value as TaskType;
                setForm((prev) => ({
                  ...prev,
                  task_type,
                  recommended_context_length:
                    task_type === "embedding" || task_type === "rerank"
                      ? 8192
                      : prev.recommended_context_length,
                }));
              }}
            >
              <option value="chat">LLM（生成・チャット）</option>
              <option value="embedding">Embedding（ベクトル化）</option>
              <option value="rerank">Reranker（スコア / rerank）</option>
            </select>
          </Field>
          {(form.task_type === "embedding" || form.task_type === "rerank") && (
            <>
              <Field label="推奨コンテキスト長" hint="起動時の `--max-model-len` の目安。多くの embedding / rerank モデルは 8192 以下です">
                <input
                  type="number"
                  min={1024}
                  className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
                  value={form.recommended_context_length}
                  onChange={(e) =>
                    setForm({ ...form, recommended_context_length: Number.parseInt(e.target.value, 10) || 8192 })
                  }
                />
              </Field>
              {form.task_type === "embedding" && (
                <Field label="出力次元（任意）" hint="既知なら登録（例: 1792）。smoke test で実測値と照合できます">
                  <input
                    className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
                    placeholder="例: 1792"
                    value={form.output_dimension}
                    onChange={(e) => setForm({ ...form, output_dimension: e.target.value })}
                  />
                </Field>
              )}
              <Field label="ライセンス注意（任意）" hint="NonCommercial など運用上の注意をメモできます">
                <input
                  className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2"
                  placeholder="例: NonCommercial License"
                  value={form.license_note}
                  onChange={(e) => setForm({ ...form, license_note: e.target.value })}
                />
              </Field>
            </>
          )}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4 text-sm text-gray-400">
          <label className="flex items-start gap-2 bg-bg-tertiary/60 border border-white/5 rounded-lg p-3">
            <input type="checkbox" checked={form.gated} onChange={(e) => setForm({ ...form, gated: e.target.checked })} />
            <span className="flex items-center gap-1 text-white">
              利用許諾が必要なモデル
              <InfoTooltip text="Hugging Face の gated model の場合にオンにします。.env の HF_TOKEN と、Hugging Face 側での利用許諾が必要です。" />
            </span>
          </label>
          <label className="flex items-start gap-2 bg-bg-tertiary/60 border border-white/5 rounded-lg p-3">
            <input type="checkbox" checked={form.trust_remote_code} onChange={(e) => setForm({ ...form, trust_remote_code: e.target.checked })} />
            <span className="flex items-center gap-1 text-white">
              trust_remote_code を許可
              <InfoTooltip text="モデルが独自 Python コードを要求する場合だけオンにします。信頼できるモデル以外では有効にしないでください。" />
            </span>
          </label>
        </div>
        <button
          onClick={registerModel}
          disabled={!form.id}
          className="mt-4 px-4 py-2 bg-accent-primary text-white rounded-lg disabled:opacity-50"
        >
          モデル候補として登録
        </button>
        {message && <p className="mt-3 text-sm text-gray-400">{message}</p>}
      </div>

      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
          <Download className="w-5 h-5 text-accent-success" />
          2. 登録済みモデルをダウンロード
          <InfoTooltip text="vLLM を起動する前に、モデルを Hugging Face から共有キャッシュへ保存します。大きなモデルは時間がかかります。" />
        </h2>
        <div className="mb-3 flex items-center justify-between gap-2">
          <p className="text-xs text-gray-500">
            未ダウンロードモデルは「一覧から削除（表示のみ）」で非表示にできます（モデルデータは残ります）。
          </p>
          {hiddenModelIds.length > 0 && (
            <button
              onClick={resetHiddenModels}
              className="px-3 py-1.5 text-xs bg-bg-primary/80 border border-white/10 text-gray-300 rounded-lg"
            >
              非表示をリセット ({hiddenModelIds.length})
            </button>
          )}
        </div>
        <div className="mb-4 flex flex-wrap gap-2">
          {MODEL_TABS.map((tab) => {
            const count = models.filter(
              (model) => !hiddenModelIds.includes(model.id) && modelMatchesTab(model, tab.key)
            ).length;
            const active = modelTab === tab.key;
            return (
              <button
                key={tab.key}
                type="button"
                onClick={() => setModelTab(tab.key)}
                className={`px-3 py-1.5 text-sm rounded-lg border ${
                  active
                    ? "bg-accent-primary/20 border-accent-primary/50 text-accent-primary"
                    : "bg-bg-primary/60 border-white/10 text-gray-400 hover:text-white"
                }`}
              >
                {tab.label}
                <span className="ml-1.5 text-xs opacity-80">{count}</span>
              </button>
            );
          })}
        </div>
        <div className="space-y-3">
          {models.length === 0 && (
            <div className="bg-bg-tertiary rounded-lg p-4 text-sm text-gray-400">
              まだモデルが登録されていません。上のフォームで Hugging Face の repo_id を登録してください。
            </div>
          )}
          {models.filter((model) => !hiddenModelIds.includes(model.id) && modelMatchesTab(model, modelTab)).length === 0 &&
            models.length > 0 && (
            <div className="bg-bg-tertiary rounded-lg p-4 text-sm text-gray-400">
              このタブに表示するモデルはありません。
            </div>
          )}
          {models.filter((model) => !hiddenModelIds.includes(model.id) && modelMatchesTab(model, modelTab)).map((model) => {
            const now = Date.now() / 1000;
            const activeJob = jobs.find((job) => job.model_id === model.id && ["queued", "running"].includes(job.status));
            const activeJobs = jobs.filter(
              (job) => job.model_id === model.id && ["queued", "running"].includes(job.status)
            );
            const latestCompletedJob = jobs
              .filter((job) => job.model_id === model.id && job.status === "completed")
              .sort((a, b) => b.updated_at - a.updated_at)[0];
            const latestFailedJob = jobs
              .filter((job) => job.model_id === model.id && job.status === "failed")
              .sort((a, b) => b.updated_at - a.updated_at)[0];
            const showFailedJob =
              Boolean(latestFailedJob) &&
              (!latestCompletedJob || latestFailedJob.updated_at > latestCompletedJob.updated_at);
            const activeIdleSeconds = activeJob ? Math.max(0, now - activeJob.updated_at) : 0;
            const isLikelyStalled =
              Boolean(activeJob) &&
              activeIdleSeconds >= 90 &&
              (jobSpeedBytesPerSec[activeJob?.id ?? ""] || 0) <= 0;
            const isMaybeSlow = Boolean(activeJob) && activeIdleSeconds >= 45 && !isLikelyStalled;
            const activeProgress = activeJob ? jobProgressPercent(activeJob) : 0;
            const displayStatus = activeJob
              ? `${activeJob.status === "queued" ? "待機中" : "ダウンロード中"}: ${jobProgressLabel(activeJob)}`
              : model.downloaded
                ? `ダウンロード済み ${formatBytes(model.cache_size_bytes)}`
                : "未ダウンロード";
            return (
              <div key={model.id} className="bg-bg-tertiary rounded-lg p-4">
                <div className="min-w-0">
                  <p className="font-medium">{model.name}</p>
                  <p className="text-xs text-gray-500 font-mono">{model.id}</p>
                  <p className="text-xs text-gray-500">
                    {model.size} / {taskTypeLabel(model.task_type)} / {displayStatus}
                    {model.task_type === "embedding" && model.output_dimension
                      ? ` / ${model.output_dimension}d`
                      : ""}
                  </p>
                  <div className="mt-2 flex items-center gap-2">
                    <label className="text-xs text-gray-500 whitespace-nowrap" htmlFor={`task-type-${model.id}`}>
                      用途
                    </label>
                    <select
                      id={`task-type-${model.id}`}
                      className="bg-bg-primary border border-white/10 rounded-lg px-2 py-1 text-xs text-gray-200"
                      value={model.task_type || "chat"}
                      disabled={Boolean(activeJob)}
                      onChange={(e) => updateTaskType(model, e.target.value as TaskType)}
                    >
                      <option value="chat">LLM</option>
                      <option value="embedding">Embedding</option>
                      <option value="rerank">Reranker</option>
                    </select>
                  </div>
                  {activeJob && (
                    <div className="mt-2">
                      <div className="h-2 bg-bg-primary rounded-full overflow-hidden">
                        <div
                          className={`h-full bg-accent-primary ${activeProgress === 0 && activeJob.status === "running" ? "animate-pulse w-1/3" : ""}`}
                          style={{
                            width: activeProgress > 0 ? `${activeProgress}%` : activeJob.status === "running" ? "33%" : "0%",
                          }}
                        />
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {activeJob.status === "queued" ? "待機中" : "ダウンロード中"}: {jobProgressLabel(activeJob)}
                      </p>
                      <p className="text-xs text-gray-500">
                        速度: {formatSpeed(jobSpeedBytesPerSec[activeJob.id] || 0)}
                      </p>
                      <p className="text-xs text-gray-500">
                        最終更新: {formatElapsed(activeIdleSeconds)}前
                      </p>
                      {activeJobs.length > 1 && (
                        <p className="text-xs text-accent-warning">
                          実行中ジョブが {activeJobs.length} 件あります（重複）。キャンセルして再開を推奨します。
                        </p>
                      )}
                    </div>
                  )}
                  {isMaybeSlow && (
                    <div className="mt-2 rounded-md border border-white/10 bg-bg-primary/60 p-2">
                      <p className="text-xs text-gray-400">
                        進捗の更新が少し遅いです。大きなファイルは一時的に止まって見えることがあります。
                      </p>
                    </div>
                  )}
                  {isLikelyStalled && (
                    <div className="mt-2 rounded-md border border-accent-warning/40 bg-accent-warning/10 p-2 space-y-2">
                      <p className="text-xs text-accent-warning font-medium">
                        進捗が {formatElapsed(activeIdleSeconds)}止まっています
                      </p>
                      <p className="text-xs text-gray-300">
                        対処: 「続きから再開」を押すと、ダウンロード済み部分を再利用して再開します。
                        90秒以上止まった場合はサーバー側でも自動再開を試みます（最大3回）。
                      </p>
                      <button
                        onClick={() => cancelAndResume(model.id)}
                        disabled={resumingModelId === model.id}
                        className="px-3 py-2 text-xs bg-accent-warning/20 border border-accent-warning/40 text-accent-warning rounded-lg disabled:opacity-50"
                      >
                        {resumingModelId === model.id ? "再開中..." : "続きから再開"}
                      </button>
                    </div>
                  )}
                  {!activeJob && latestCompletedJob && (
                    <p className="mt-1 text-xs text-gray-500">
                      最終完了: {new Date(latestCompletedJob.updated_at * 1000).toLocaleString()}
                    </p>
                  )}
                  {!activeJob && showFailedJob && latestFailedJob && (
                    <div className="mt-2 rounded-md border border-accent-danger/30 bg-accent-danger/10 p-2">
                      <p className="text-xs font-medium text-accent-danger">
                        ダウンロード失敗
                      </p>
                      <p className="text-xs text-gray-300">
                        {latestFailedJob.error_hint || latestFailedJob.message || "原因不明のエラー"}
                      </p>
                    </div>
                  )}
                </div>
                <div className="mt-3 flex flex-wrap gap-2 md:justify-end">
                  <button
                    onClick={() => startDownload(model.id)}
                    disabled={Boolean(activeJob)}
                    className="px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg disabled:opacity-50"
                  >
                    {activeJob ? "処理中..." : model.downloaded ? "再ダウンロード" : "このモデルをダウンロード"}
                  </button>
                  {activeJob && !isLikelyStalled && (
                    <button
                      onClick={() => cancelAndResume(model.id)}
                      disabled={resumingModelId === model.id}
                      className="px-3 py-2 text-xs bg-bg-primary/80 border border-white/10 text-gray-300 rounded-lg disabled:opacity-50"
                    >
                      {resumingModelId === model.id ? "再開中..." : "続きから再開"}
                    </button>
                  )}
                  {!model.downloaded && (
                    <button
                      onClick={() => removeFromVisibleList(model.id)}
                      disabled={Boolean(activeJob)}
                      className="inline-flex items-center gap-1 px-3 py-2 text-xs bg-bg-primary/80 border border-white/10 text-gray-300 rounded-lg disabled:opacity-50"
                    >
                      <EyeOff className="w-3.5 h-3.5" />
                      一覧から削除（表示のみ）
                    </button>
                  )}
                  <button
                    onClick={() => removeModel(model.id)}
                    disabled={Boolean(activeJob)}
                    className="px-3 py-2 text-xs bg-accent-danger/20 border border-accent-danger/40 text-accent-danger rounded-lg disabled:opacity-50"
                  >
                    完全削除（一覧＋キャッシュ）
                  </button>
                </div>
              </div>
            );
          })}
        </div>
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
