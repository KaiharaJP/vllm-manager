"use client";

import { useState, useEffect } from "react";
import ServerControl from "@/components/ServerControl";
import MetricsPanel from "@/components/MetricsPanel";
import ConfigPanel from "@/components/ConfigPanel";
import LogPanel from "@/components/LogPanel";
import Header from "@/components/Header";
import ModelManagement from "@/components/ModelManagement";
import LiteLLMAdminPanel from "@/components/LiteLLMAdminPanel";
import UserManagementPanel from "@/components/UserManagementPanel";
import UsagePanel from "@/components/UsagePanel";
import SystemOverview from "@/components/SystemOverview";
import { api } from "@/lib/api";
import type { AppUser, DownloadJob, ServerStatus, Model, ServerConfig, ContextPreset } from "@/types";

interface DashboardProps {
  currentUser: AppUser;
  onLogout: () => void;
}

type TabKey = "overview" | "control" | "metrics" | "models" | "users" | "litellm" | "usage" | "config" | "log";
const TAB_STORAGE_KEY = "vllm_manager_active_tab";

export default function Dashboard({ currentUser, onLogout }: DashboardProps) {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [jobs, setJobs] = useState<DownloadJob[]>([]);
  const [contextPresets, setContextPresets] = useState<ContextPreset[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>("overview");

  useEffect(() => {
    const saved = window.localStorage.getItem(TAB_STORAGE_KEY) as TabKey | null;
    if (!saved) return;
    const allowedTabs: TabKey[] = currentUser.role === "admin"
      ? ["overview", "control", "metrics", "models", "users", "litellm", "usage", "config", "log"]
      : ["overview", "metrics", "users"];
    setActiveTab(allowedTabs.includes(saved) ? saved : "overview");
  }, [currentUser.role]);

  useEffect(() => {
    window.localStorage.setItem(TAB_STORAGE_KEY, activeTab);
  }, [activeTab]);

  useEffect(() => {
    loadInitialData();
    const statusInterval = setInterval(refreshStatus, 10000);
    const jobsInterval =
      currentUser.role === "admin" ? setInterval(refreshModels, 5000) : null;
    return () => {
      clearInterval(statusInterval);
      if (jobsInterval) clearInterval(jobsInterval);
    };
  }, []);

  /** API が返らない場合でも画面を永久ブロックしない */
  useEffect(() => {
    if (!loading) return;
    const t = window.setTimeout(() => setLoading(false), 32_000);
    return () => window.clearTimeout(t);
  }, [loading]);

  async function loadInitialData() {
    try {
      setLoadError(null);
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
      setLoadError(err instanceof Error ? err.message : String(err));
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
    try {
      const [modelsData, jobsData] = await Promise.all([
        api.getModels(),
        currentUser.role === "admin" ? api.getModelDownloads() : Promise.resolve([]),
      ]);
      setModels(modelsData);
      setJobs(jobsData);
    } catch {
      /* バックグラウンド更新は失敗しても画面全体は維持 */
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
      <Header status={status} user={currentUser} onLogout={onLogout} />

      {loadError && (
        <div className="bg-accent-danger/15 border-b border-accent-danger/40 px-4 py-3 text-sm text-red-200">
          <p className="font-medium">初期データの取得に失敗しました</p>
          <p className="text-red-300/90 mt-1">{loadError}</p>
        </div>
      )}

      {/* タブナビゲーション */}
      <nav className="sticky top-0 z-10 bg-bg-secondary/80 backdrop-blur-sm border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            {([
              { key: "overview", label: "ホーム" },
              ...(currentUser.role === "admin" ? [
                { key: "control", label: "サーバー管理" },
              ] as const : []),
              { key: "metrics", label: "モニタリング" },
              ...(currentUser.role !== "admin" ? [
                { key: "users", label: "マイページ" },
              ] as const : []),
              ...(currentUser.role === "admin" ? [
                { key: "models", label: "モデル管理" },
                { key: "users", label: "ユーザー管理" },
                { key: "litellm", label: "APIキー/チーム" },
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
        {activeTab === "overview" && <SystemOverview />}
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
        {activeTab === "users" && <UserManagementPanel currentUser={currentUser} />}
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
