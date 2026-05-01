"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type {
  ServerStatus,
  ServerConfig,
  Model,
  ContextPreset,
  ServerStartRequest,
  ApiResponse,
} from "@/types";
import { Play, StopCircle, RotateCcw, Terminal } from "lucide-react";

interface ServerControlProps {
  status: ServerStatus | null;
  config: ServerConfig | null;
  models: Model[];
  contextPresets: ContextPreset[];
  onActionComplete: () => void;
}

export default function ServerControl({
  status,
  config,
  models,
  contextPresets,
  onActionComplete,
}: ServerControlProps) {
  const [form, setForm] = useState({
    model_id: config?.model_id ?? models[0]?.id ?? "",
    context_length: config?.context_length ?? 8192,
    max_num_seqs: config?.max_num_seqs ?? 256,
    gpu_memory_utilization: config?.gpu_memory_utilization ?? 0.9,
    tensor_parallel_size: config?.tensor_parallel_size ?? 1,
    download_model: true,
  });

  const [action, setAction] = useState<"idle" | "starting" | "stopping" | "restarting">("idle");
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [showSteps, setShowSteps] = useState(false);

  async function handleStart() {
    setAction("starting");
    setResult(null);
    setShowSteps(true);
    try {
      const res = await api.startServer(form as ServerStartRequest);
      setResult(res);
      onActionComplete();
    } catch (err) {
      setResult({ success: false, message: String(err) });
    } finally {
      setAction("idle");
    }
  }

  async function handleStop() {
    setAction("stopping");
    try {
      const res = await api.stopServer();
      setResult(res);
      onActionComplete();
    } catch (err) {
      setResult({ success: false, message: String(err) });
    } finally {
      setAction("idle");
    }
  }

  async function handleRestart() {
    setAction("restarting");
    setResult(null);
    setShowSteps(true);
    try {
      const res = await api.restartServer();
      setResult(res);
      onActionComplete();
    } catch (err) {
      setResult({ success: false, message: String(err) });
    } finally {
      setAction("idle");
    }
  }

  const isBusy = action !== "idle";

  return (
    <div className="space-y-6 animate-slide-in">
      {/* 操作パネル */}
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Terminal className="w-5 h-5 text-accent-primary" />
          サーバー操作
        </h2>

        {/* モデル選択 */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">モデル</label>
          <select
            value={form.model_id}
            onChange={(e) => setForm({ ...form, model_id: e.target.value })}
            disabled={isBusy}
            className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-primary disabled:opacity-50"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.size})
              </option>
            ))}
          </select>
        </div>

        {/* コンテキスト長 */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">コンテキスト長</label>
          <div className="flex gap-2 flex-wrap">
            {contextPresets.map((p) => (
              <button
                key={p.value}
                onClick={() => setForm({ ...form, context_length: p.value })}
                disabled={isBusy}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  form.context_length === p.value
                    ? "bg-accent-primary text-white"
                    : "bg-bg-tertiary text-gray-400 hover:text-white border border-white/10"
                } disabled:opacity-50`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* スロット数 */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">
            最大同時リクエスト数: {form.max_num_seqs}
          </label>
          <input
            type="range"
            min="1"
            max="512"
            step="1"
            value={form.max_num_seqs}
            onChange={(e) => setForm({ ...form, max_num_seqs: parseInt(e.target.value) })}
            disabled={isBusy}
            className="w-full accent-accent-primary"
          />
        </div>

        {/* GPU メモリ利用率 */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">
            GPU メモリ利用率: {Math.round(form.gpu_memory_utilization * 100)}%
          </label>
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={form.gpu_memory_utilization}
            onChange={(e) =>
              setForm({ ...form, gpu_memory_utilization: parseFloat(e.target.value) })
            }
            disabled={isBusy}
            className="w-full accent-accent-primary"
          />
        </div>

        {/* テンソル並列数 */}
        <div className="mb-4">
          <label className="block text-sm text-gray-400 mb-1">
            テンソル並列数: {form.tensor_parallel_size}
          </label>
          <input
            type="range"
            min="1"
            max="8"
            step="1"
            value={form.tensor_parallel_size}
            onChange={(e) => setForm({ ...form, tensor_parallel_size: parseInt(e.target.value) })}
            disabled={isBusy}
            className="w-full accent-accent-primary"
          />
        </div>

        {/* ダウンロード制御 */}
        <label className="mb-6 flex items-center gap-2 text-sm text-gray-400">
          <input
            type="checkbox"
            checked={form.download_model}
            onChange={(e) => setForm({ ...form, download_model: e.target.checked })}
            disabled={isBusy}
          />
          起動前にモデルキャッシュを確認/ダウンロードする
        </label>

        {/* ボタン */}
        <div className="flex gap-3">
          <button
            onClick={handleStart}
            disabled={isBusy}
            className="flex items-center gap-2 px-6 py-2.5 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            {action === "starting" ? "起動中..." : "起動"}
          </button>

          <button
            onClick={handleStop}
            disabled={isBusy || !status?.running}
            className="flex items-center gap-2 px-6 py-2.5 bg-accent-danger/20 hover:bg-accent-danger/30 text-accent-danger rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <StopCircle className="w-4 h-4" />
            {action === "stopping" ? "停止中..." : "停止"}
          </button>

          <button
            onClick={handleRestart}
            disabled={isBusy || !status?.running}
            className="flex items-center gap-2 px-6 py-2.5 bg-bg-tertiary hover:bg-bg-tertiary/80 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed border border-white/10"
          >
            <RotateCcw className="w-4 h-4" />
            {action === "restarting" ? "再起動中..." : "再起動"}
          </button>
        </div>
      </div>

      {/* 実行結果 */}
      {result && showSteps && (
        <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                result.success ? "bg-accent-success" : "bg-accent-danger"
              }`}
            />
            {result.success ? "成功" : "エラー"}
          </h3>
          <div className="bg-bg-primary rounded-lg p-3 font-mono text-xs text-gray-300 space-y-1">
            {result.steps?.map((step, i) => (
              <div key={i} className="animate-slide-in" style={{ animationDelay: `${i * 100}ms` }}>
                <span className="text-accent-primary">{">"}</span> {step}
              </div>
            ))}
            {result.message && !result.steps?.length && (
              <div>{result.message}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
