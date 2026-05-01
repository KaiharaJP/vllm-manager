"use client";

import type { ServerStatus } from "@/types";
import { Activity, Power, Shield } from "lucide-react";

interface HeaderProps {
  status: ServerStatus | null;
}

export default function Header({ status }: HeaderProps) {
  const isHealthy = status?.healthy ?? false;
  const isRunning = status?.running ?? false;

  return (
    <header className="border-b border-white/5 bg-bg-secondary/50">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8 text-accent-primary" />
          <div>
            <h1 className="text-xl font-bold">vLLM Manager</h1>
            <p className="text-xs text-gray-500">Server Management Dashboard</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* ステータスインジケーター */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-bg-tertiary">
            <div
              className={`w-2.5 h-2.5 rounded-full ${
                isHealthy
                  ? "bg-accent-success animate-pulse"
                  : isRunning
                  ? "bg-accent-warning animate-pulse"
                  : "bg-gray-600"
              }`}
            />
            <span className="text-sm text-gray-300">
              {isHealthy ? "Healthy" : isRunning ? "Starting..." : "Stopped"}
            </span>
          </div>

          {/* モデル名 */}
          {status?.model && (
            <div className="text-sm text-gray-400 font-mono truncate max-w-[200px]">
              {status.model}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
