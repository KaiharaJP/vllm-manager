"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { MetricsData, MetricsMessage } from "@/types";

const WS_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useMetricsWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [history, setHistory] = useState<MetricsData[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = WS_BASE.startsWith("https") ? "wss" : "ws";
    const baseUrl = WS_BASE.replace(/https?:\/\//, "");
    const ws = new WebSocket(`${protocol}://${baseUrl}/ws/metrics`);

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const msg: MetricsMessage = JSON.parse(event.data);

        if (msg.type === "history" && Array.isArray(msg.data)) {
          setHistory(msg.data as MetricsData[]);
          setMetrics((msg.data as MetricsData[]).at(-1) ?? null);
        } else if (msg.type === "metrics") {
          setMetrics(msg.data as MetricsData);
          setHistory((prev) => [...prev.slice(-99), msg.data as MetricsData]);
        } else if (msg.type === "error") {
          setError(msg.message ?? "Unknown error");
        }
      } catch {
        // 無効なメッセージは無視
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setTimeout(connect, 5000); // 5秒後に再接続
    };

    ws.onerror = () => {
      setError("WebSocket connection failed");
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return { metrics, history, connected, error };
}
