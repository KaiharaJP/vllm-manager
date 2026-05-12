"use client";

import { useMetricsWebSocket } from "@/hooks/useMetricsWebSocket";
import type { LiteLLMProxyRequestRow } from "@/types";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Cpu, Zap, Clock, Database, Eye, Wrench, Copy, ChevronRight } from "lucide-react";

export default function MetricsPanel() {
  const { metrics, history, litellmProxyRequests, connected, error } = useMetricsWebSocket();

  if (error) {
    return (
      <div className="bg-bg-secondary rounded-xl border border-accent-danger/30 p-6 text-center">
        <p className="text-accent-danger mb-2">接続エラー</p>
        <p className="text-sm text-gray-400">{error}</p>
        <p className="text-xs text-gray-500 mt-2">vLLM サーバーが起動しているか確認してください</p>
      </div>
    );
  }

  if (!connected) {
    return (
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6 text-center">
        <div className="w-8 h-8 border-2 border-accent-primary border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-gray-400">WebSocket 接続中...</p>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6 text-center">
        <p className="text-gray-400">メトリクスデータを受信待ち...</p>
      </div>
    );
  }

  function formatUptime(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
  }

  const sortedLiteLLM = [...litellmProxyRequests].sort((a, b) => b.started_at - a.started_at);

  return (
    <div className="space-y-6 animate-slide-in">
      <LiteLLMRequestMonitor rows={sortedLiteLLM} />

      {/* サマリーカード */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={<Cpu className="w-5 h-5" />}
          label="GPU メモリ"
          value={`${metrics.gpu_memory_usage_gb.toFixed(1)} GB`}
          sub={`${metrics.kv_cache_usage_perc.toFixed(1)}% KV Cache`}
          color="text-accent-primary"
        />
        <MetricCard
          icon={<Zap className="w-5 h-5" />}
          label="リクエスト"
          value={String(metrics.num_requests_running)}
          sub={`${metrics.num_requests_waiting} waiting`}
          color="text-accent-success"
        />
        <MetricCard
          icon={<Clock className="w-5 h-5" />}
          label="トークン/秒"
          value={String(metrics.iteration_tokens)}
          sub={`${metrics.time_per_output_token_ms.toFixed(1)}ms/token`}
          color="text-accent-warning"
        />
        <MetricCard
          icon={<Database className="w-5 h-5" />}
          label="スループット"
          value={`${metrics.request_throughput_rps.toFixed(2)} rps`}
          sub={`GPU: ${metrics.gpu_compute_time.toFixed(2)}s`}
          color="text-accent-secondary"
        />
      </div>

      {/* グラフ */}
      {history.length > 1 && (
        <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-4">GPU メモリ使用量 (GB)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="colorMem" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="timestamp"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(ts) => new Date(ts * 1000).toLocaleTimeString()}
                stroke="#4a4a6a"
              />
              <YAxis stroke="#4a4a6a" />
              <Tooltip
                contentStyle={{
                  background: "#1a1a2e",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  color: "#e2e8f0",
                }}
                labelFormatter={(ts) => new Date(ts * 1000).toLocaleTimeString()}
              />
              <Area
                type="monotone"
                dataKey="gpu_memory_usage_gb"
                stroke="#6366f1"
                fill="url(#colorMem)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* リアルタイム詳細 */}
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h3 className="text-sm font-medium text-gray-400 mb-4">詳細メトリクス</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
          <DetailRow label="KV Cache 使用率" value={`${metrics.kv_cache_usage_perc.toFixed(1)}%`} />
          <DetailRow label="GPU 計算時間" value={`${metrics.gpu_compute_time.toFixed(3)}s`} />
          <DetailRow label="平均トークン時間" value={`${metrics.time_per_output_token_ms.toFixed(1)}ms`} />
          <DetailRow label="キュー待ちリクエスト" value={String(metrics.num_requests_waiting)} />
          <DetailRow label="スワップ中リクエスト" value={String(metrics.num_requests_swapped)} />
          <DetailRow label="GPU メモリ合計" value={`${metrics.gpu_cpu_total_gb.toFixed(1)} GB`} />
        </div>
      </div>
    </div>
  );
}

function MetricCard({ icon, label, value, sub, color }: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
  color: string;
}) {
  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 p-4">
      <div className={`mb-2 ${color}`}>{icon}</div>
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-xl font-bold">{value}</p>
      <p className="text-xs text-gray-500 mt-1">{sub}</p>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-bg-tertiary rounded-lg px-3 py-2">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="font-mono text-sm">{value}</p>
    </div>
  );
}

function LiteLLMRequestMonitor({ rows }: { rows: LiteLLMProxyRequestRow[] }) {
  const activeRows = rows.filter((r) => r.status === "streaming" || r.status === "pending");
  const latest = rows[0];
  const modelLabel = latest?.model || "vllm-local";

  function formatToken(value: number | null): string {
    if (value == null) return "—";
    return value.toLocaleString();
  }

  const statusTone =
    activeRows.length > 0
      ? "border-accent-success/40 bg-accent-success/10 text-accent-success"
      : "border-white/20 bg-white/5 text-gray-300";

  return (
    <div>
      <h3 className="text-2xl font-semibold mb-4">Loaded Models</h3>
      <div className="bg-bg-secondary rounded-2xl border border-accent-primary/40 p-4 md:p-5">
        <div className="flex items-center justify-between">
          <span className={`inline-flex items-center rounded-md border px-3 py-1 text-sm font-semibold ${statusTone}`}>
            {activeRows.length > 0 ? "READY" : "IDLE"}
          </span>
          <button
            type="button"
            className="rounded-lg p-2 text-gray-400 hover:bg-white/5 hover:text-gray-200"
            aria-label="model details"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <div className="inline-flex items-center rounded-lg border border-white/10 bg-bg-tertiary px-3 py-2 text-sm">
            <span className="font-mono text-gray-400 mr-2">llm</span>
            <span className="font-semibold text-accent-primary">{modelLabel}</span>
          </div>
          <button type="button" className="rounded-lg border border-amber-400/50 p-2 text-amber-300 hover:bg-amber-400/10">
            <Eye className="w-4 h-4" />
          </button>
          <button type="button" className="rounded-lg border border-accent-primary/60 p-2 text-accent-primary hover:bg-accent-primary/10">
            <Wrench className="w-4 h-4" />
          </button>
          <button type="button" className="rounded-lg border border-white/15 p-2 text-gray-300 hover:bg-white/10">
            <Copy className="w-4 h-4" />
          </button>
          <div className="inline-flex items-center rounded-lg border border-white/15 bg-bg-tertiary px-3 py-2 text-sm text-gray-200">
            <span className="font-semibold">CURL</span>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap gap-2 text-sm">
          <div className="rounded-lg border border-white/15 bg-bg-tertiary px-3 py-2 text-gray-200">
            Active {activeRows.length}
          </div>
          <div className="rounded-lg border border-white/15 bg-bg-tertiary px-3 py-2 text-gray-200">
            Parallel {Math.max(1, activeRows.length)}
          </div>
          <div className="rounded-lg border border-white/15 bg-bg-tertiary px-3 py-2 text-gray-200">
            Tracked {rows.length}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          {(activeRows.length > 0 ? activeRows : rows.slice(0, 2)).map((r) => (
            <div key={r.id} className="rounded-xl border border-white/10 bg-bg-tertiary p-3">
              <div className="flex items-center justify-between">
                <p className="font-mono text-xs text-gray-400">{r.id.slice(0, 8)}…</p>
                <span
                  className={`text-xs font-semibold ${
                    r.status === "error"
                      ? "text-accent-danger"
                      : r.status === "completed"
                        ? "text-gray-400"
                        : "text-accent-success"
                  }`}
                >
                  {r.status.toUpperCase()}
                </span>
              </div>
              <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                <div className="rounded-md bg-bg-primary px-2 py-1">
                  <p className="text-gray-500">prompt</p>
                  <p className="font-mono text-sm">{formatToken(r.prompt_tokens)}</p>
                </div>
                <div className="rounded-md bg-bg-primary px-2 py-1">
                  <p className="text-gray-500">completion</p>
                  <p className="font-mono text-sm">{formatToken(r.completion_tokens)}</p>
                </div>
                <div className="rounded-md bg-bg-primary px-2 py-1">
                  <p className="text-gray-500">total</p>
                  <p className="font-mono text-sm">{formatToken(r.total_tokens)}</p>
                </div>
              </div>
              {r.error && <p className="mt-2 text-xs text-accent-danger truncate">{r.error}</p>}
            </div>
          ))}
          {rows.length === 0 && (
            <div className="rounded-xl border border-dashed border-white/20 bg-bg-tertiary p-4 text-sm text-gray-400 md:col-span-2">
              LiteLLM 経由リクエスト待機中。`/v1/chat/completions` を実行するとここに同時リクエストごとの token が表示されます。
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
