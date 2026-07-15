"use client";

import { useState } from "react";
import type { AppUser, ServerStatus } from "@/types";
import { Activity, Bell, Shield } from "lucide-react";

interface HeaderProps {
  status: ServerStatus | null;
  user: AppUser;
  onLogout: () => void;
}

export default function Header({ status, user, onLogout }: HeaderProps) {
  const isHealthy = status?.healthy ?? false;
  const isRunning = status?.running ?? false;
  const [warningsOpen, setWarningsOpen] = useState(false);
  const securityWarnings = user.security_warnings ?? [];
  const showBell = user.role === "admin" && securityWarnings.length > 0;

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
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Shield className="w-4 h-4" />
            <span>{user.username} / {user.role}</span>
          </div>

          {showBell && (
            <div className="relative">
              <button
                onClick={() => setWarningsOpen((prev) => !prev)}
                className="relative p-2 rounded-full hover:bg-white/5 text-amber-300"
                title="セキュリティ上の推奨事項"
              >
                <Bell className="w-5 h-5" />
                <span className="absolute -top-0.5 -right-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-accent-warning text-[10px] font-semibold text-black">
                  {securityWarnings.length}
                </span>
              </button>

              {warningsOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setWarningsOpen(false)}
                  />
                  <div className="absolute right-0 top-full mt-2 w-80 z-20 bg-bg-secondary border border-accent-warning/40 rounded-xl shadow-lg p-4">
                    <p className="text-sm font-medium text-amber-100 mb-2">
                      セキュリティ上の推奨事項
                    </p>
                    <ul className="list-disc list-inside space-y-1 text-sm text-amber-200/90">
                      {securityWarnings.map((warning, idx) => (
                        <li key={idx}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                </>
              )}
            </div>
          )}

          <button
            onClick={onLogout}
            className="text-sm text-gray-400 hover:text-white"
          >
            Logout
          </button>
        </div>
      </div>
    </header>
  );
}
