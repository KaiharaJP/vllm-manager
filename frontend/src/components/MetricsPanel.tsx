"use client";

import { useMetricsWebSocket } from "@/hooks/useMetricsWebSocket";
import type { LiteLLMProxyRequestRow, MetricsData } from "@/types";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Cpu, Zap, Layers, Activity } from "lucide-react";

const chartTooltipStyle = {
  background: "#1a1a2e",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: "8px",
  color: "#e2e8f0",
};

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString();
}

function formatTokS(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value >= 100) return `${Math.round(value)}`;
  return value.toFixed(1);
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

function m(n: MetricsData, key: keyof MetricsData, fallback = 0): number {
  const v = n[key];
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

export default function MetricsPanel() {
  const { metrics, history, litellmProxyRequests, connected, connectionError, scrapeError } =
    useMetricsWebSocket();

  if (connectionError) {
    return (
      <div className="bg-bg-secondary rounded-xl border border-accent-danger/30 p-6 text-center">
        <p className="text-accent-danger mb-2">接続エラー</p>
        <p className="text-sm text-gray-400">{connectionError}</p>
        <p className="text-xs text-gray-500 mt-2">WebSocket に接続できません。ページを再読み込みしてください。</p>
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
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6 text-center space-y-3">
        <p className="text-gray-400">メトリクスデータを受信待ち...</p>
        {scrapeError && <p className="text-xs text-amber-400/90">{scrapeError}</p>}
        <p className="text-xs text-gray-500">vLLM 起動後 5〜10 秒で表示されます</p>
      </div>
    );
  }

  const sortedLiteLLM = [...litellmProxyRequests].sort((a, b) => b.started_at - a.started_at);
  const litellmActive = sortedLiteLLM.filter((r) => r.status === "streaming" || r.status === "pending").length;
  const vllmRunning = metrics.num_requests_running;
  const queueMismatch = Math.abs(litellmActive - vllmRunning) > 2 && (litellmActive > 0 || vllmRunning > 0);

  const decodeTokSEstimate =
    metrics.time_per_output_token_ms > 0 ? 1000 / metrics.time_per_output_token_ms : 0;

  const promptTokS = m(metrics, "prompt_throughput_tok_s");
  const genTokS = m(metrics, "generation_throughput_tok_s");
  const waitCapacity = m(metrics, "num_requests_waiting_capacity");
  const waitDeferred = m(metrics, "num_requests_waiting_deferred");

  return (
    <div className="space-y-6 animate-slide-in">
      {scrapeError && (
        <p className="text-xs text-amber-400/90 bg-amber-400/10 border border-amber-400/30 rounded-lg px-3 py-2">
          vLLM メトリクス取得: {scrapeError}（vLLM 停止中は表示されません。起動後 5〜10 秒お待ちください）
        </p>
      )}
      <EngineQueueSection metrics={metrics} waitCapacity={waitCapacity} waitDeferred={waitDeferred} />

      {queueMismatch && (
        <p className="text-xs text-amber-400/90 bg-amber-400/10 border border-amber-400/30 rounded-lg px-3 py-2">
          LiteLLM 追跡中 {litellmActive} 本と vLLM Running {vllmRunning} 本に差があります。直叩き（:18000）や
          ヘッダ未付きリクエストは LiteLLM 表に出ませんが、vLLM キューには含まれます。
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={<Layers className="w-5 h-5" />}
          label="Running"
          value={String(metrics.num_requests_running)}
          sub={`Waiting ${metrics.num_requests_waiting} (cap ${waitCapacity} / def ${waitDeferred})`}
          color="text-accent-success"
        />
        <MetricCard
          icon={<Activity className="w-5 h-5" />}
          label="Prefill tok/s"
          value={formatTokS(promptTokS)}
          sub="エンジン合計（5s 差分）"
          color="text-accent-warning"
        />
        <MetricCard
          icon={<Zap className="w-5 h-5" />}
          label="Generation tok/s"
          value={formatTokS(genTokS)}
          sub={
            decodeTokSEstimate > 0
              ? `目安 ~${formatTokS(decodeTokSEstimate)} tok/s/本 (${metrics.time_per_output_token_ms.toFixed(0)}ms/tok)`
              : "エンジン合計（5s 差分）"
          }
          color="text-accent-primary"
        />
        <MetricCard
          icon={<Cpu className="w-5 h-5" />}
          label="GPU メモリ"
          value={`${metrics.gpu_memory_usage_gb.toFixed(1)} GB`}
          sub={`${metrics.kv_cache_usage_perc.toFixed(1)}% KV Cache`}
          color="text-accent-secondary"
        />
      </div>

      {history.length > 1 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartCard title="リクエストキュー">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={history}>
                <XAxis
                  dataKey="timestamp"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tickFormatter={formatTime}
                  stroke="#4a4a6a"
                />
                <YAxis stroke="#4a4a6a" allowDecimals={false} />
                <Tooltip contentStyle={chartTooltipStyle} labelFormatter={formatTime} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="num_requests_running"
                  name="Running"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="num_requests_waiting"
                  name="Waiting"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="エンジンスループット (tok/s)">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={history}>
                <XAxis
                  dataKey="timestamp"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tickFormatter={formatTime}
                  stroke="#4a4a6a"
                />
                <YAxis stroke="#4a4a6a" />
                <Tooltip contentStyle={chartTooltipStyle} labelFormatter={formatTime} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="prompt_throughput_tok_s"
                  name="Prefill"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="generation_throughput_tok_s"
                  name="Generation"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}

      <LiteLLMRequestTable rows={sortedLiteLLM} vllmRunning={vllmRunning} />

      {history.length > 1 && (
        <ChartCard title="GPU メモリ使用量 (GB)">
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
                tickFormatter={formatTime}
                stroke="#4a4a6a"
              />
              <YAxis stroke="#4a4a6a" />
              <Tooltip contentStyle={chartTooltipStyle} labelFormatter={formatTime} />
              <Area
                type="monotone"
                dataKey="gpu_memory_usage_gb"
                stroke="#6366f1"
                fill="url(#colorMem)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h3 className="text-sm font-medium text-gray-400 mb-4">詳細メトリクス</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
          <DetailRow label="KV Cache 使用率" value={`${metrics.kv_cache_usage_perc.toFixed(1)}%`} />
          <DetailRow label="平均デコード時間" value={`${metrics.time_per_output_token_ms.toFixed(1)} ms/tok`} />
          <DetailRow label="リクエスト RPS" value={metrics.request_throughput_rps.toFixed(2)} />
          <DetailRow label="Waiting (capacity)" value={String(waitCapacity)} />
          <DetailRow label="Waiting (deferred)" value={String(waitDeferred)} />
          <DetailRow label="スワップ中" value={String(metrics.num_requests_swapped)} />
          <DetailRow label="GPU メモリ合計" value={`${metrics.gpu_cpu_total_gb.toFixed(1)} GB`} />
          <DetailRow label="GPU 計算時間" value={`${metrics.gpu_compute_time.toFixed(3)} s`} />
          <DetailRow label="LiteLLM 追跡中" value={String(litellmActive)} />
        </div>
      </div>
    </div>
  );
}

function EngineQueueSection({
  metrics,
  waitCapacity,
  waitDeferred,
}: {
  metrics: MetricsData;
  waitCapacity: number;
  waitDeferred: number;
}) {
  return (
    <div className="bg-bg-secondary rounded-xl border border-accent-primary/30 p-5">
      <h3 className="text-sm font-medium text-gray-400 mb-4">Engine Queue（vLLM）</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <QueueStat label="Running" value={metrics.num_requests_running} accent="text-accent-success" />
        <QueueStat label="Waiting" value={metrics.num_requests_waiting} accent="text-amber-400" />
        <QueueStat label="Waiting (capacity)" value={waitCapacity} accent="text-orange-300" />
        <QueueStat label="Waiting (deferred)" value={waitDeferred} accent="text-gray-300" />
      </div>
    </div>
  );
}

function QueueStat({ label, value, accent }: { label: string; value: number; accent: string }) {
  return (
    <div className="bg-bg-tertiary rounded-lg px-4 py-3 text-center">
      <p className="text-xs text-gray-500">{label}</p>
      <p className={`text-3xl font-bold font-mono mt-1 ${accent}`}>{value}</p>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
      <h3 className="text-sm font-medium text-gray-400 mb-4">{title}</h3>
      {children}
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  sub,
  color,
}: {
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

function LiteLLMRequestTable({ rows, vllmRunning }: { rows: LiteLLMProxyRequestRow[]; vllmRunning: number }) {
  const activeRows = rows.filter((r) => r.status === "streaming" || r.status === "pending");
  const displayRows = [
    ...activeRows,
    ...rows.filter((r) => r.status !== "streaming" && r.status !== "pending"),
  ].slice(0, 20);

  function formatToken(value: number | null): string {
    if (value == null) return "—";
    return value.toLocaleString();
  }

  function phaseLabel(r: LiteLLMProxyRequestRow): string {
    const phase = r.phase ?? (r.status === "pending" ? "prefill" : "done");
    if (phase === "prefill") return "prefill…";
    return phase;
  }

  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 p-5">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <h3 className="text-sm font-medium text-gray-400">LiteLLM リクエスト（X-Vllm-Manager-Source: litellm）</h3>
        <div className="flex gap-2 text-xs">
          <span className="rounded-md border border-white/15 bg-bg-tertiary px-2 py-1 text-gray-300">
            Active {activeRows.length}
          </span>
          <span className="rounded-md border border-white/15 bg-bg-tertiary px-2 py-1 text-gray-300">
            vLLM Running {vllmRunning}
          </span>
        </div>
      </div>

      {displayRows.length === 0 ? (
        <p className="text-sm text-gray-400 border border-dashed border-white/20 rounded-lg p-4">
          LiteLLM 経由リクエスト待機中。chat completions を実行すると、リクエストごとのフェーズと tok/s が表示されます。
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-500 border-b border-white/10">
                <th className="pb-2 pr-3 font-medium">ID</th>
                <th className="pb-2 pr-3 font-medium">status</th>
                <th className="pb-2 pr-3 font-medium">phase</th>
                <th className="pb-2 pr-3 font-medium text-right">elapsed</th>
                <th className="pb-2 pr-3 font-medium text-right">prompt</th>
                <th className="pb-2 pr-3 font-medium text-right">completion</th>
                <th className="pb-2 pr-3 font-medium text-right">prefill tok/s</th>
                <th className="pb-2 pr-3 font-medium text-right">gen tok/s</th>
              </tr>
            </thead>
            <tbody>
              {displayRows.map((r) => (
                <tr key={r.id} className="border-b border-white/5 hover:bg-white/[0.02]">
                  <td className="py-2 pr-3 font-mono text-xs text-gray-400">{r.id.slice(0, 8)}…</td>
                  <td className="py-2 pr-3">
                    <StatusBadge status={r.status} />
                  </td>
                  <td className="py-2 pr-3 text-gray-300">{phaseLabel(r)}</td>
                  <td className="py-2 pr-3 text-right font-mono">{formatElapsed(r.elapsed_s ?? 0)}</td>
                  <td className="py-2 pr-3 text-right font-mono">{formatToken(r.prompt_tokens)}</td>
                  <td className="py-2 pr-3 text-right font-mono">{formatToken(r.completion_tokens)}</td>
                  <td className="py-2 pr-3 text-right font-mono">
                    {(r.phase ?? "") === "prefill" && r.first_token_at == null
                      ? "—"
                      : formatTokS(r.prefill_tok_s)}
                  </td>
                  <td className="py-2 pr-3 text-right font-mono">{formatTokS(r.gen_tok_s)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: LiteLLMProxyRequestRow["status"] }) {
  const tone =
    status === "error"
      ? "text-accent-danger"
      : status === "completed"
        ? "text-gray-400"
        : "text-accent-success";
  return <span className={`text-xs font-semibold ${tone}`}>{status}</span>;
}
