"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { AppEvent, DownloadJob, LiteLLMProxyRequestRow, MetricsData, MetricsMessage } from "@/types";

const WS_BASE = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");

export function useMetricsWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [history, setHistory] = useState<MetricsData[]>([]);
  const [events, setEvents] = useState<AppEvent[]>([]);
  const [downloads, setDownloads] = useState<DownloadJob[]>([]);
  const [litellmProxyRequests, setLitellmProxyRequests] = useState<LiteLLMProxyRequestRow[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mergeLiteLLMRow = useCallback((row: LiteLLMProxyRequestRow) => {
    setLitellmProxyRequests((prev) => {
      const m = new Map(prev.map((r) => [r.id, r]));
      m.set(row.id, row);
      const now = Date.now() / 1000;
      return Array.from(m.values()).filter((r) => {
        if (r.status === "streaming" || r.status === "pending") return true;
        return now - r.updated_at < 15;
      });
    });
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const wsUrl = (() => {
      if (WS_BASE) {
        const protocol = WS_BASE.startsWith("https") ? "wss" : "ws";
        const baseUrl = WS_BASE.replace(/https?:\/\//, "");
        return `${protocol}://${baseUrl}/ws/metrics`;
      }
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      return `${protocol}://${window.location.host}/ws/metrics`;
    })();
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as MetricsMessage | AppEvent;

        if (msg.type === "event_history" && Array.isArray((msg as AppEvent).data)) {
          const arr = (msg as AppEvent).data as AppEvent[];
          setEvents(arr);
          const litellmRows = arr
            .filter((e) => e.type === "litellm_proxy_request" && e.data && typeof e.data === "object")
            .map((e) => e.data as LiteLLMProxyRequestRow);
          if (litellmRows.length > 0) {
            const m = new Map<string, LiteLLMProxyRequestRow>();
            for (const r of litellmRows) m.set(r.id, r);
            const now = Date.now() / 1000;
            setLitellmProxyRequests(
              Array.from(m.values()).filter((r) => {
                if (r.status === "streaming" || r.status === "pending") return true;
                return now - r.updated_at < 15;
              }),
            );
          }
        } else if (msg.type === "litellm_proxy_snapshot" && msg.data && typeof msg.data === "object") {
          const reqs = (msg.data as { requests?: LiteLLMProxyRequestRow[] }).requests;
          if (Array.isArray(reqs)) setLitellmProxyRequests(reqs);
        } else if (msg.type === "history" && Array.isArray(msg.data)) {
          setHistory(msg.data as MetricsData[]);
          setMetrics((msg.data as MetricsData[]).at(-1) ?? null);
        } else if (msg.type === "metrics") {
          const data = (msg as AppEvent).data as MetricsData;
          setMetrics(data);
          setHistory((prev) => [...prev.slice(-99), data]);
          setEvents((prev) => [...prev.slice(-199), msg as AppEvent]);
        } else if ((msg as AppEvent).type === "model_download") {
          const job = (msg as AppEvent).data as DownloadJob;
          setDownloads((prev) => {
            const next = prev.filter((item) => item.id !== job.id);
            return [job, ...next].slice(0, 50);
          });
          setEvents((prev) => [...prev.slice(-199), msg as AppEvent]);
        } else if (msg.type === "error") {
          setError(msg.message ?? "Unknown error");
          setEvents((prev) => [...prev.slice(-199), msg as AppEvent]);
        } else if (msg.type === "litellm_proxy_request" && msg.data && typeof msg.data === "object") {
          mergeLiteLLMRow(msg.data as LiteLLMProxyRequestRow);
          setEvents((prev) => [...prev.slice(-199), msg as AppEvent]);
        } else if (msg.type !== "pong") {
          setEvents((prev) => [...prev.slice(-199), msg as AppEvent]);
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
  }, [mergeLiteLLMRow]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return { metrics, history, events, downloads, litellmProxyRequests, connected, error };
}
