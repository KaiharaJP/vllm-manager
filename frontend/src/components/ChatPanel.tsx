"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MessageSquare, RotateCcw, Send, Square } from "lucide-react";
import { api, streamChatCompletion } from "@/lib/api";
import type { ChatUiMessage } from "@/types";

const CHAT_HISTORY_KEY = "vllm_manager_chat_history";
const CHAT_MODEL_KEY = "vllm_manager_chat_model";

function loadStoredHistory(): ChatUiMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(CHAT_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (m): m is ChatUiMessage =>
        m != null &&
        typeof m === "object" &&
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string"
    );
  } catch {
    return [];
  }
}

function loadStoredModel(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(CHAT_MODEL_KEY) || "";
}

export default function ChatPanel() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [messages, setMessages] = useState<ChatUiMessage[]>([]);
  const [input, setInput] = useState("");
  const [loadingModels, setLoadingModels] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const refreshModels = useCallback(async () => {
    try {
      setLoadingModels(true);
      const data = await api.getChatModels();
      setModels(data.models);
      const stored = loadStoredModel();
      if (stored && data.models.includes(stored)) {
        setSelectedModel(stored);
      } else if (data.models.length > 0) {
        setSelectedModel(data.models[0]);
      } else {
        setSelectedModel("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingModels(false);
    }
  }, []);

    useEffect(() => {
    setMessages(loadStoredHistory());
    refreshModels();
    const interval = window.setInterval(refreshModels, 30_000);
    return () => window.clearInterval(interval);
  }, [refreshModels]);

  useEffect(() => {
    window.localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    if (selectedModel) {
      window.localStorage.setItem(CHAT_MODEL_KEY, selectedModel);
    }
  }, [selectedModel]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  function handleNewConversation() {
    if (sending) return;
    setMessages([]);
    setError(null);
    setInput("");
  }

  function handleStop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setSending(false);
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || sending || !selectedModel) return;

    const userMessage: ChatUiMessage = { role: "user", content: text };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    setError(null);
    setSending(true);

    const assistantMessage: ChatUiMessage = { role: "assistant", content: "" };
    setMessages([...nextMessages, assistantMessage]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChatCompletion({
        model: selectedModel,
        messages: nextMessages,
        signal: controller.signal,
        onDelta: (delta) => {
          setMessages((prev) => {
            if (prev.length === 0) return prev;
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role !== "assistant") return prev;
            updated[updated.length - 1] = {
              ...last,
              content: last.content + delta,
            };
            return updated;
          });
        },
      });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "assistant" && !last.content) {
          return updated.slice(0, -1);
        }
        return updated;
      });
    } finally {
      abortRef.current = null;
      setSending(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  }

  return (
    <div className="space-y-6 animate-slide-in">
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6 flex flex-col min-h-[70vh]">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-accent-primary" />
            チャット
          </h2>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleNewConversation}
              disabled={sending}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-lg border border-white/10 text-gray-300 hover:bg-white/5 disabled:opacity-50"
            >
              <RotateCcw className="w-4 h-4" />
              新しい会話
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 mb-4">
          <label className="text-sm text-gray-400" htmlFor="chat-model">
            モデル（稼働中の vLLM）
          </label>
          <select
            id="chat-model"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={loadingModels || sending || models.length === 0}
            className="bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-accent-primary min-w-[280px] max-w-full"
          >
            {models.length === 0 ? (
              <option value="">利用可能なモデルがありません</option>
            ) : (
              models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))
            )}
          </select>
          {loadingModels && <span className="text-xs text-gray-500">読み込み中...</span>}
        </div>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-accent-danger/10 border border-accent-danger/30 text-sm text-red-200">
            {error}
          </div>
        )}

        {!loadingModels && models.length === 0 && (
          <div className="mb-4 p-4 rounded-lg bg-bg-tertiary border border-white/10 text-sm text-gray-400">
            現在チャット用に稼働中の vLLM がありません。管理者にモデルの起動を依頼するか、しばらく待ってから再読み込みしてください。
          </div>
        )}

        <div className="flex-1 overflow-y-auto space-y-3 mb-4 pr-1 min-h-[320px] max-h-[52vh]">
          {messages.length === 0 ? (
            <p className="text-sm text-gray-500 text-center py-12">
              メッセージを入力して会話を始めてください。Shift+Enter で改行できます。
            </p>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm whitespace-pre-wrap ${
                    msg.role === "user"
                      ? "bg-accent-primary/20 border border-accent-primary/30 text-white"
                      : "bg-bg-tertiary border border-white/10 text-gray-100"
                  }`}
                >
                  {msg.content || (sending && idx === messages.length - 1 ? "…" : "")}
                </div>
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>

        <div className="flex gap-2 items-end border-t border-white/5 pt-4">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={sending || !selectedModel}
            rows={3}
            placeholder={
              selectedModel
                ? "メッセージを入力…（Enter で送信、Shift+Enter で改行）"
                : "モデルが選択できるまでお待ちください"
            }
            className="flex-1 bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-sm resize-none focus:ring-2 focus:ring-accent-primary disabled:opacity-50"
          />
          {sending ? (
            <button
              type="button"
              onClick={handleStop}
              className="inline-flex items-center gap-1.5 px-4 py-2.5 bg-accent-danger/80 hover:bg-accent-danger text-white rounded-lg text-sm"
            >
              <Square className="w-4 h-4" />
              停止
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void handleSend()}
              disabled={!input.trim() || !selectedModel}
              className="inline-flex items-center gap-1.5 px-4 py-2.5 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-lg text-sm disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
              送信
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
