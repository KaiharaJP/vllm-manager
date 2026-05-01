"use client";

import { useState, useEffect } from "react";
import ServerControl from "@/components/ServerControl";
import MetricsPanel from "@/components/MetricsPanel";
import ConfigPanel from "@/components/ConfigPanel";
import LogPanel from "@/components/LogPanel";
import Header from "@/components/Header";
import { api } from "@/lib/api";
import type { ServerStatus, Model, ServerConfig, ContextPreset } from "@/types";

export default function Dashboard() {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [contextPresets, setContextPresets] = useState<ContextPreset[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"control" | "metrics" | "config" | "log">("control");

  useEffect(() => {
    loadInitialData();
    const interval = setInterval(refreshStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  async function loadInitialData() {
    try {
      const [statusData, configData, modelsData, presetsData] = await Promise.all([
        api.getStatus(),
        api.getConfig(),
        api.getModels(),
        api.getContextPresets(),
      ]);
      setStatus(statusData);
      setConfig(configData);
      setModels(modelsData);
      setContextPresets(presetsData);
    } catch (err) {
      console.error("Failed to load initial data:", err);
    } finally {
      setLoading(false);
    }
  }

  async function refreshStatus() {
    try {
      const data = await api.getStatus();
      setStatus(data);
    } catch {
      //  silently fail - status polling
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-accent-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-primary">
      <Header status={status} />

      {/* タブナビゲーション */}
      <nav className="sticky top-0 z-10 bg-bg-secondary/80 backdrop-blur-sm border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            {([
              { key: "control", label: "サーバー管理" },
              { key: "metrics", label: "モニタリング" },
              { key: "config", label: "設定" },
              { key: "log", label: "ログ" },
            ] as const).map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`px-4 py-3 text-sm font-medium transition-colors border-b-2 ${
                  activeTab === tab.key
                    ? "border-accent-primary text-accent-primary"
                    : "border-transparent text-gray-400 hover:text-white"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* コンテンツ */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === "control" && (
          <ServerControl
            status={status}
            config={config}
            models={models}
            contextPresets={contextPresets}
            onActionComplete={refreshStatus}
          />
        )}
        {activeTab === "metrics" && <MetricsPanel />}
        {activeTab === "config" && (
          <ConfigPanel
            config={config}
            models={models}
            contextPresets={contextPresets}
          />
        )}
        {activeTab === "log" && <LogPanel />}
      </main>
    </div>
  );
}
