"use client";

import { useMetricsWebSocket } from "@/hooks/useMetricsWebSocket";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Cpu, Zap, Clock, Database } from "lucide-react";

export default function MetricsPanel() {
  const { metrics, history, connected, error } = useMetricsWebSocket();

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

  return (
    <div className="space-y-6 animate-slide-in">
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
