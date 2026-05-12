"use client";

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Cpu, HardDrive, MemoryStick, Monitor } from "lucide-react";
import { api } from "@/lib/api";
import type { SystemMetrics } from "@/types";

export default function SystemOverview() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const data = await api.getSystemMetrics();
        if (mounted) {
          setMetrics(data);
          setError(null);
        }
      } catch (err) {
        if (mounted) setError(String(err));
      }
    };
    load();
    const timer = setInterval(load, 5000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  if (!metrics && !error) {
    return <p className="text-sm text-gray-400">システム情報を取得中...</p>;
  }

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
          <Monitor className="w-5 h-5 text-accent-primary" />
          システム監視（ホーム）
        </h2>
        <p className="text-sm text-gray-400">
          このPCでの CPU / メモリ / SSD / GPU の現在使用率です。5秒ごとに更新されます。
        </p>
      </div>

      {error && (
        <div className="bg-accent-danger/10 border border-accent-danger/30 rounded-xl p-4 text-sm text-accent-danger">
          システム情報の取得に失敗しました: {error}
        </div>
      )}

      {metrics && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <MetricCard
              icon={<Cpu className="w-4 h-4 text-accent-primary" />}
              title="CPU"
              percent={metrics.cpu.usage_percent}
              detail={`${metrics.cpu.cores_logical} threads`}
            />
            <MetricCard
              icon={<MemoryStick className="w-4 h-4 text-accent-success" />}
              title="メモリ"
              percent={metrics.memory.usage_percent}
              detail={`${metrics.memory.used_gb} / ${metrics.memory.total_gb} GB`}
            />
            <MetricCard
              icon={<HardDrive className="w-4 h-4 text-accent-warning" />}
              title="SSD (/)"
              percent={metrics.disk.usage_percent}
              detail={`${metrics.disk.used_gb} / ${metrics.disk.total_gb} GB`}
            />
          </div>

          <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
            <h3 className="text-sm font-medium text-gray-300 mb-3">GPU 使用率</h3>
            {metrics.gpus.length === 0 ? (
              <p className="text-sm text-gray-500">GPU情報を取得できませんでした（nvidia-smi 未検出の可能性）。</p>
            ) : (
              <div className="space-y-3">
                {metrics.gpus.map((gpu) => {
                  const memPercent =
                    gpu.memory_total_mb > 0
                      ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100
                      : 0;
                              const gpuProcesses = metrics.gpu_processes.filter(
                                (proc) => proc.gpu_index === gpu.index
                              );
                  return (
                    <div key={gpu.index} className="bg-bg-tertiary rounded-lg p-3">
                      <p className="text-sm font-medium">
                        GPU {gpu.index}: {gpu.name}
                      </p>
                      <p className="text-xs text-gray-500 mb-2">
                        温度 {gpu.temperature_c}C / VRAM {gpu.memory_used_mb.toFixed(0)}MB /{" "}
                        {gpu.memory_total_mb.toFixed(0)}MB
                      </p>
                      <div className="space-y-2">
                        <Bar label="GPU 使用率" percent={gpu.utilization_percent} />
                        <Bar label="VRAM 使用率" percent={memPercent} />
                      </div>
                                  <div className="mt-3">
                                    <p className="text-xs text-gray-400 mb-1">
                                      動作中プロセス（このGPU）
                                    </p>
                                    {gpuProcesses.length === 0 ? (
                                      <p className="text-xs text-gray-500">プロセスはありません</p>
                                    ) : (
                                      <div className="space-y-1">
                                        {gpuProcesses.map((proc) => (
                                          <div
                                            key={`${proc.gpu_uuid}-${proc.pid}`}
                                            className="text-xs text-gray-300 border border-white/5 rounded px-2 py-1"
                                          >
                                            <span className="font-mono">PID {proc.pid}</span>{" "}
                                            {proc.process_name} - {proc.used_memory_mb.toFixed(0)}MB
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function MetricCard({
  icon,
  title,
  percent,
  detail,
}: {
  icon: ReactNode;
  title: string;
  percent: number;
  detail: string;
}) {
  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 p-4">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <p className="text-sm font-medium">{title}</p>
      </div>
      <Bar label={`${percent.toFixed(1)}%`} percent={percent} />
      <p className="text-xs text-gray-500 mt-2">{detail}</p>
    </div>
  );
}

function Bar({ label, percent }: { label: string; percent: number }) {
  const safe = Math.max(0, Math.min(100, percent));
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
      </div>
      <div className="h-2 rounded-full bg-bg-primary overflow-hidden">
        <div className="h-full bg-accent-primary" style={{ width: `${safe}%` }} />
      </div>
    </div>
  );
}
