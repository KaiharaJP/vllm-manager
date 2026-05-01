"use client";

import { useState, useEffect } from "react";
import ServerControl from "@/components/ServerControl";
import MetricsPanel from "@/components/MetricsPanel";
import ConfigPanel from "@/components/ConfigPanel";
import LogPanel from "@/components/LogPanel";
import Header from "@/components/Header";
import ModelManagement from "@/components/ModelManagement";
import LiteLLMAdminPanel from "@/components/LiteLLMAdminPanel";
import UsagePanel from "@/components/UsagePanel";
import { api } from "@/lib/api";
import type { AppUser, DownloadJob, ServerStatus, Model, ServerConfig, ContextPreset } from "@/types";

interface DashboardProps {
  currentUser: AppUser;
  onLogout: () => void;
}

type TabKey = "control" | "metrics" | "models" | "litellm" | "usage" | "config" | "log";

export default function Dashboard({ currentUser, onLogout }: DashboardProps) {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [jobs, setJobs] = useState<DownloadJob[]>([]);
  const [contextPresets, setContextPresets] = useState<ContextPreset[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabKey>(currentUser.role === "admin" ? "control" : "metrics");

  useEffect(() => {
    loadInitialData();
    const interval = setInterval(refreshStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  async function loadInitialData() {
    try {
      const [statusData, configData, modelsData, presetsData, jobsData] = await Promise.all([
        api.getStatus(),
        api.getConfig(),
        api.getModels(),
        api.getContextPresets(),
        currentUser.role === "admin" ? api.getModelDownloads() : Promise.resolve([]),
      ]);
      setStatus(statusData);
      setConfig(configData);
      setModels(modelsData);
      setContextPresets(presetsData);
      setJobs(jobsData);
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

  async function refreshModels() {
    const [modelsData, jobsData] = await Promise.all([
      api.getModels(),
      currentUser.role === "admin" ? api.getModelDownloads() : Promise.resolve([]),
    ]);
    setModels(modelsData);
    setJobs(jobsData);
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
      <Header status={status} user={currentUser} onLogout={onLogout} />

      {/* タブナビゲーション */}
      <nav className="sticky top-0 z-10 bg-bg-secondary/80 backdrop-blur-sm border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            {([
              ...(currentUser.role === "admin" ? [
                { key: "control", label: "サーバー管理" },
              ] as const : []),
              { key: "metrics", label: "モニタリング" },
              ...(currentUser.role === "admin" ? [
                { key: "models", label: "モデル管理" },
                { key: "litellm", label: "ユーザー/APIキー" },
                { key: "usage", label: "利用状況" },
                { key: "config", label: "設定" },
                { key: "log", label: "ログ" },
              ] as const : []),
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
        {activeTab === "control" && currentUser.role === "admin" && (
          <ServerControl
            status={status}
            config={config}
            models={models}
            contextPresets={contextPresets}
            onActionComplete={refreshStatus}
          />
        )}
        {activeTab === "metrics" && <MetricsPanel />}
        {activeTab === "models" && currentUser.role === "admin" && (
          <ModelManagement models={models} jobs={jobs} onChanged={refreshModels} />
        )}
        {activeTab === "litellm" && currentUser.role === "admin" && <LiteLLMAdminPanel />}
        {activeTab === "usage" && currentUser.role === "admin" && <UsagePanel />}
        {activeTab === "config" && currentUser.role === "admin" && (
          <ConfigPanel
            config={config}
            models={models}
            contextPresets={contextPresets}
          />
        )}
        {activeTab === "log" && currentUser.role === "admin" && <LogPanel />}
      </main>
    </div>
  );
}
