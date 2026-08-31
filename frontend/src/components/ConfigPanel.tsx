"use client";

import { useState, type ReactNode } from "react";
import type { ServerConfig, Model, ContextPreset } from "@/types";
import { Info, Key, Settings } from "lucide-react";

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
        <p className="text-xs text-gray-500 mb-4">
          「現在設定」は、最後にサーバー起動時に使われた設定値です。次回「サーバー管理」画面を開いたときの初期値にも使われます。各項目の（i）に、vLLM / OpenAI API との対応関係を載せています。
        </p>

        {config && (
          <div className="space-y-3">
            <ConfigRow
              label="モデル"
              value={config.model_id}
              hint="Hugging Face のモデル ID。`vllm serve` の引数になります。"
            />
            <ConfigRow
              label="コンテキスト長"
              value={String(config.context_length)}
              hint="vLLM `--max-model-len`。KV キャッシュ規模に強く効く上限トークンです。"
            />
            <ConfigRow
              label="最大同時リクエスト"
              value={String(config.max_num_seqs)}
              hint="vLLM `--max-num-seqs`。同時処理スロット数の上限です。"
            />
            <ConfigRow
              label="デフォルト max_tokens"
              value={String(config.default_max_tokens)}
              hint="プロキシがリクエストに `max_tokens` が無いとき注入する値。"
            />
            <ConfigRow
              label="デフォルト temperature"
              value={String(config.default_temperature)}
              hint="未指定時に注入する sampling temperature。"
            />
            <ConfigRow
              label="デフォルト top_p"
              value={String(config.default_top_p)}
              hint="未指定時に注入する nucleus sampling の p。"
            />
            <ConfigRow
              label="デフォルト frequency_penalty"
              value={String(config.default_frequency_penalty)}
              hint="未指定時に注入。繰り返し抑制（OpenAI 互換）。"
            />
            <ConfigRow
              label="デフォルト presence_penalty"
              value={String(config.default_presence_penalty)}
              hint="未指定時に注入。話題の広がり（OpenAI 互換）。"
            />
            <ConfigRow
              label="GPU メモリ設定モード"
              value={
                config.gpu_memory_mode === "auto"
                  ? "自動"
                  : config.gpu_memory_mode === "minimal"
                    ? "最低限"
                    : "手動"
              }
              hint="自動=空きVRAMを最大確保 / 最低限=context長×同時実行数から必要分のみ確保 / 手動=固定割合"
            />
            <ConfigRow
              label="GPU メモリ利用率"
              value={`${Math.round(config.gpu_memory_utilization * 100)}%`}
              hint="vLLM `--gpu-memory-utilization`。KV 等に使える VRAM 比率の上限イメージです。"
            />
            <ConfigRow
              label="テンソル並列数"
              value={String(config.tensor_parallel_size)}
              hint="vLLM `--tensor-parallel-size`。モデルを分割して載せる GPU 枚数です。"
            />
            <ConfigRow
              label="使用GPU（保存値）"
              value={config.gpu_devices || "all"}
              hint="起動時に `CUDA_VISIBLE_DEVICES` へ渡した値の保存です。"
            />
            <ConfigRow
              label="vLLM ポート"
              value={String(config.vllm_port)}
              hint="コンテナ内で vLLM が待ち受けるポート。このアプリが管理するプロセスの `--port` に対応します。"
            />
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

function ConfigRow({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="flex justify-between items-start gap-3 py-2 border-b border-white/5 last:border-0">
      <span className="text-sm text-gray-400 flex items-start gap-1 shrink-0">
        {label}
        <InfoTooltip text={hint} />
      </span>
      <span className="text-sm font-mono text-right break-all max-w-[min(100%,20rem)]">{value}</span>
    </div>
  );
}

function InfoTooltip({ text }: { text: ReactNode }) {
  return (
    <span className="group relative inline-flex shrink-0">
      <Info className="h-4 w-4 cursor-help text-gray-500 transition-colors group-hover:text-accent-primary mt-0.5" />
      <span className="pointer-events-none absolute left-0 top-6 z-30 hidden w-72 rounded-lg border border-white/10 bg-bg-primary p-3 text-xs font-normal leading-relaxed text-gray-300 shadow-xl group-hover:block">
        {text}
      </span>
    </span>
  );
}
