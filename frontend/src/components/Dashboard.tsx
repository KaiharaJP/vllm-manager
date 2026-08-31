"use client";

import { useState, useEffect } from "react";
import ServerControl from "@/components/ServerControl";
import MetricsPanel from "@/components/MetricsPanel";
import RequestHistoryPanel from "@/components/RequestHistoryPanel";
import ConfigPanel from "@/components/ConfigPanel";
import LogPanel from "@/components/LogPanel";
import Header from "@/components/Header";
import ModelManagement from "@/components/ModelManagement";
import LiteLLMAdminPanel from "@/components/LiteLLMAdminPanel";
import UserManagementPanel from "@/components/UserManagementPanel";
import UsagePanel from "@/components/UsagePanel";
import StoragePanel from "@/components/StoragePanel";
import SystemOverview from "@/components/SystemOverview";
import ChatPanel from "@/components/ChatPanel";
import { api } from "@/lib/api";
import type { AppUser, DownloadJob, ServerStatus, Model, ServerConfig, ContextPreset } from "@/types";

interface DashboardProps {
  currentUser: AppUser;
  onLogout: () => void;
}

type TabKey = "overview" | "chat" | "control" | "metrics" | "requestHistory" | "models" | "users" | "litellm" | "usage" | "storage" | "config" | "log";
const TAB_STORAGE_KEY = "vllm_manager_active_tab";

export default function Dashboard({ currentUser, onLogout }: DashboardProps) {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [jobs, setJobs] = useState<DownloadJob[]>([]);
  const [contextPresets, setContextPresets] = useState<ContextPreset[]>([]);
  const [statusLoading, setStatusLoading] = useState(true);
  const [secondaryLoading, setSecondaryLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>("overview");

  useEffect(() => {
    const saved = window.localStorage.getItem(TAB_STORAGE_KEY) as TabKey | null;
    if (!saved) return;
    const allowedTabs: TabKey[] = currentUser.role === "admin"
      ? ["overview", "chat", "control", "metrics", "requestHistory", "models", "users", "litellm", "usage", "storage", "config", "log"]
      : ["overview", "chat", "metrics", "users"];
    setActiveTab(allowedTabs.includes(saved) ? saved : "overview");
  }, [currentUser.role]);

  useEffect(() => {
    window.localStorage.setItem(TAB_STORAGE_KEY, activeTab);
  }, [activeTab]);

  useEffect(() => {
    void loadInitialData();
    const statusInterval = setInterval(refreshStatus, 10000);
    const jobsInterval =
      currentUser.role === "admin" ? setInterval(refreshModels, 5000) : null;
    return () => {
      clearInterval(statusInterval);
      if (jobsInterval) clearInterval(jobsInterval);
    };
  }, [currentUser.role]);

  async function loadSecondaryData() {
    setSecondaryLoading(true);
    try {
      const results = await Promise.allSettled([
        api.getConfig(),
        api.getModels(),
        api.getContextPresets(),
        currentUser.role === "admin" ? api.getModelDownloads() : Promise.resolve([]),
      ]);
      const [configResult, modelsResult, presetsResult, jobsResult] = results;
      if (configResult.status === "fulfilled") setConfig(configResult.value);
      if (modelsResult.status === "fulfilled") setModels(modelsResult.value);
      if (presetsResult.status === "fulfilled") setContextPresets(presetsResult.value);
      if (jobsResult.status === "fulfilled") setJobs(jobsResult.value);
      const failed = results.find((r) => r.status === "rejected");
      if (failed?.status === "rejected") {
        const reason = failed.reason;
        setLoadError(reason instanceof Error ? reason.message : String(reason));
      }
    } finally {
      setSecondaryLoading(false);
    }
  }

  async function loadInitialData() {
    setStatusLoading(true);
    void loadSecondaryData();
    try {
      setLoadError(null);
      const statusData = await api.getStatus();
      setStatus(statusData);
    } catch (err) {
      console.error("Failed to load status:", err);
      setLoadError(err instanceof Error ? err.message : String(err));
    } finally {
      setStatusLoading(false);
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

  return (
    <div className="min-h-screen bg-bg-primary">
      <Header status={status} user={currentUser} onLogout={onLogout} />

      {statusLoading && (
        <div className="bg-bg-secondary/80 border-b border-white/5 px-4 py-2 text-xs text-gray-500">
          サーバー状態を読み込み中...
        </div>
      )}

      {loadError && (
        <div className="bg-accent-danger/15 border-b border-accent-danger/40 px-4 py-3 text-sm text-red-200">
          <p className="font-medium">データの取得に失敗しました</p>
          <p className="text-red-300/90 mt-1">{loadError}</p>
        </div>
      )}

      {secondaryLoading && (
        <div className="bg-bg-secondary/80 border-b border-white/5 px-4 py-2 text-xs text-gray-500">
          モデル一覧などの追加データを読み込み中...
        </div>
      )}

      {/* タブナビゲーション */}
      <nav className="sticky top-0 z-10 bg-bg-secondary/80 backdrop-blur-sm border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1">
            {([
              { key: "overview", label: "ホーム" },
              { key: "chat", label: "チャット" },
              ...(currentUser.role === "admin" ? [
                { key: "control", label: "サーバー管理" },
              ] as const : []),
              { key: "metrics", label: "モニタリング" },
              ...(currentUser.role === "admin" ? [
                { key: "requestHistory", label: "リクエスト履歴" },
              ] as const : []),
              ...(currentUser.role !== "admin" ? [
                { key: "users", label: "マイページ" },
              ] as const : []),
              ...(currentUser.role === "admin" ? [
                { key: "models", label: "モデル管理" },
                { key: "users", label: "ユーザー管理" },
                { key: "litellm", label: "APIキー/チーム" },
                { key: "usage", label: "利用状況" },
                { key: "storage", label: "ストレージ" },
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
        {activeTab === "chat" && <ChatPanel />}
        {activeTab === "control" && currentUser.role === "admin" && (
          <ServerControl
            status={status}
            config={config}
            models={models}
            contextPresets={contextPresets}
            onActionComplete={refreshStatus}
          />
        )}
        {activeTab === "metrics" && <MetricsPanel currentUser={currentUser} />}
        {activeTab === "requestHistory" && currentUser.role === "admin" && (
          <RequestHistoryPanel />
        )}
        {activeTab === "models" && currentUser.role === "admin" && (
          <ModelManagement models={models} jobs={jobs} onChanged={refreshModels} />
        )}
        {activeTab === "users" && <UserManagementPanel currentUser={currentUser} />}
        {activeTab === "litellm" && currentUser.role === "admin" && <LiteLLMAdminPanel />}
        {activeTab === "usage" && currentUser.role === "admin" && <UsagePanel />}
        {activeTab === "storage" && currentUser.role === "admin" && <StoragePanel />}
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
