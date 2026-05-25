"use client";

import { useState } from "react";
import { Activity, Stethoscope } from "lucide-react";
import { api } from "@/lib/api";
import type { ServiceHealthCheckResult } from "@/types";

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full shrink-0 ${ok ? "bg-accent-success" : "bg-accent-danger"}`}
    />
  );
}

export default function ServiceHealthPanel() {
  const [result, setResult] = useState<ServiceHealthCheckResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runCheck() {
    setLoading(true);
    setError(null);
    try {
      setResult(await api.checkServiceHealth());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const litellmOk = result ? result.litellm.liveliness && result.litellm.readiness : null;

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border border-white/10 bg-bg-secondary/80 px-3 py-2 text-xs">
      <span className="flex items-center gap-1.5 text-gray-400">
        <Stethoscope className="w-3.5 h-3.5" />
        HTTP ヘルス
      </span>
      <button
        type="button"
        onClick={runCheck}
        disabled={loading}
        className="flex items-center gap-1 px-2 py-1 rounded-md bg-white/5 border border-white/10 text-gray-300 hover:bg-white/10 disabled:opacity-50"
      >
        <Activity className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
        {loading ? "確認中" : "チェック"}
      </button>
      {result && (
        <>
          <span className="flex items-center gap-1 text-gray-400" title={result.vllm.message}>
            <StatusDot ok={result.vllm.healthy} />
            vLLM
          </span>
          <span
            className="flex items-center gap-1 text-gray-400"
            title={
              litellmOk
                ? "LiteLLM: プロセス OK・受付可能"
                : `LiteLLM NG — liveliness: ${result.litellm.liveliness ? "OK" : "NG"}, readiness: ${result.litellm.readiness ? "OK" : "NG"}`
            }
          >
            <StatusDot ok={!!litellmOk} />
            LiteLLM
          </span>
          <span className="flex items-center gap-1 text-gray-400">
            <StatusDot ok={result.backend.healthy} />
            API
          </span>
          <span className="text-gray-500">{new Date(result.checked_at * 1000).toLocaleTimeString()}</span>
        </>
      )}
      {error && <span className="text-accent-danger">{error}</span>}
      {!result && !loading && !error && (
        <span className="text-gray-500">推論せず HTTP のみ（キューに載らない）</span>
      )}
    </div>
  );
}
