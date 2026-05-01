"use client";

import { useEffect, useState } from "react";
import { BarChart3 } from "lucide-react";
import { api } from "@/lib/api";

export default function UsagePanel() {
  const [status, setStatus] = useState<unknown>(null);
  const [spend, setSpend] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [statusData, spendData] = await Promise.allSettled([
          api.getLiteLLMStatus(),
          api.getLiteLLMSpend(),
        ]);
        if (statusData.status === "fulfilled") setStatus(statusData.value);
        if (spendData.status === "fulfilled") setSpend(spendData.value);
        if (spendData.status === "rejected") setError(String(spendData.reason));
      } catch (err) {
        setError(String(err));
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-accent-primary" />
          LiteLLM 利用状況
        </h2>
        <p className="text-sm text-gray-400">
          LiteLLM の spend logs / cost tracking を vLLM Manager から確認するためのビューです。
        </p>
        {error && <p className="mt-3 text-sm text-accent-warning">{error}</p>}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <JsonPanel title="LiteLLM Status" value={status} />
        <JsonPanel title="Spend Logs" value={spend} />
      </div>
    </div>
  );
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
      <h3 className="text-sm font-medium text-gray-400 mb-3">{title}</h3>
      <pre className="bg-bg-primary rounded-lg p-3 text-xs overflow-auto max-h-[600px]">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}
