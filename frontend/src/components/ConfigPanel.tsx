"use client";

import { useState } from "react";
import type { ServerConfig, Model, ContextPreset } from "@/types";
import { Settings, Key } from "lucide-react";

interface ConfigPanelProps {
  config: ServerConfig | null;
  models: Model[];
  contextPresets: ContextPreset[];
}

export default function ConfigPanel({ config, models, contextPresets }: ConfigPanelProps) {
  const [apiKey, setApiKey] = useState("");
  const [authEnabled, setAuthEnabled] = useState(false);

  return (
    <div className="space-y-6 animate-slide-in">
      {/* 現在設定 */}
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5 text-accent-primary" />
          現在設定
        </h2>

        {config && (
          <div className="space-y-3">
            <ConfigRow label="モデル" value={config.model_id} />
            <ConfigRow label="コンテキスト長" value={String(config.context_length)} />
            <ConfigRow label="最大同時リクエスト" value={String(config.max_num_seqs)} />
            <ConfigRow
              label="GPU メモリ利用率"
              value={`${Math.round(config.gpu_memory_utilization * 100)}%`}
            />
            <ConfigRow label="テンソル並列数" value={String(config.tensor_parallel_size)} />
            <ConfigRow label="vLLM ポート" value={String(config.vllm_port)} />
          </div>
        )}
      </div>

      {/* LiteLLM 認証設定 */}
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Key className="w-5 h-5 text-accent-warning" />
          LiteLLM 認証設定
        </h2>

        <div className="space-y-4">
          {/* 認証トグル */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">API 認証</p>
              <p className="text-xs text-gray-500">
                有効にすると API キーが必要なります
              </p>
            </div>
            <button
              onClick={() => setAuthEnabled(!authEnabled)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                authEnabled ? "bg-accent-primary" : "bg-gray-600"
              }`}
            >
              <div
                className={`absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform ${
                  authEnabled ? "translate-x-6" : "translate-x-0.5"
                }`}
              />
            </button>
          </div>

          {/* API キー設定 */}
          {authEnabled && (
            <div className="space-y-2">
              <label className="block text-sm text-gray-400">API キー</label>
              <div className="flex gap-2">
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-xxxxx"
                  className="flex-1 bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-primary"
                />
                <button className="px-4 py-2 bg-accent-primary text-white rounded-lg text-sm hover:bg-accent-primary/90">
                  保存
                </button>
              </div>
              <p className="text-xs text-gray-500">
                このキーは LiteLLM プロキシで認証に使用されます
              </p>
            </div>
          )}

          {/* 接続情報 */}
          <div className="mt-4 p-3 bg-bg-primary rounded-lg">
            <p className="text-xs text-gray-500 mb-2">接続情報</p>
            <div className="space-y-1 font-mono text-xs">
              <p>
                <span className="text-gray-500">Endpoint:</span>{" "}
                <span className="text-accent-primary">
                  {process.env.NEXT_PUBLIC_LITELLM_URL || "http://localhost:4000"}
                </span>
              </p>
              <p>
                <span className="text-gray-500">Model:</span>{" "}
                <span className="text-accent-success">vllm-local</span>
              </p>
              <p>
                <span className="text-gray-500">Auth:</span>{" "}
                <span className={authEnabled ? "text-accent-success" : "text-accent-warning"}>
                  {authEnabled ? "Enabled" : "Disabled (公開モード)"}
                </span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 使用例 */}
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h3 className="text-sm font-medium text-gray-400 mb-3">curl 使用例</h3>
        <pre className="bg-bg-primary rounded-lg p-3 text-xs font-mono text-gray-300 overflow-x-auto">
{authEnabled ? `curl ${process.env.NEXT_PUBLIC_LITELLM_URL || "http://localhost:4000"}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${apiKey || "YOUR_API_KEY"}" \\
  -d '{
    "model": "vllm-local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'` : `curl ${process.env.NEXT_PUBLIC_LITELLM_URL || "http://localhost:4000"}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "vllm-local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'`}
        </pre>
      </div>
    </div>
  );
}

function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-white/5 last:border-0">
      <span className="text-sm text-gray-400">{label}</span>
      <span className="text-sm font-mono">{value}</span>
    </div>
  );
}
