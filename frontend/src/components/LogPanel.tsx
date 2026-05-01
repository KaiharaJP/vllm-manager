"use client";

import { useState, useEffect, useRef } from "react";
import { api } from "@/lib/api";
import { ArrowDown } from "lucide-react";

export default function LogPanel() {
  const [log, setLog] = useState("");
  const [loading, setLoading] = useState(true);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    fetchLog();
    const interval = setInterval(fetchLog, 3000);
    return () => clearInterval(interval);
  }, []);

  async function fetchLog() {
    try {
      const data = await api.getLog(200);
      setLog(data.log);
      // 自動スクロール
      if (logRef.current) {
        logRef.current.scrollTop = logRef.current.scrollHeight;
      }
    } catch {
      // silently fail
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="animate-slide-in">
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">vLLM サーバーログ</h2>
          <button
            onClick={fetchLog}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-white"
          >
            <ArrowDown className="w-3 h-3" />
            最新へ
          </button>
        </div>

        <div className="relative">
          <pre
            ref={logRef}
            className="bg-bg-primary rounded-lg p-4 h-[500px] overflow-auto font-mono text-xs text-gray-300 whitespace-pre-wrap"
          >
            {loading ? (
              <span className="text-gray-500">ログ読み込み中...</span>
            ) : log ? (
              log
            ) : (
              <span className="text-gray-500">ログがありません</span>
            )}
          </pre>
        </div>
      </div>
    </div>
  );
}
