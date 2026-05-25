"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";
import type { ChatMessage, LiteLLMProxyRequestDetail } from "@/types";
import { promptFullText } from "@/lib/promptFingerprint";

function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part && typeof part === "object" && "text" in part) {
          return String((part as { text?: string }).text ?? "");
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  if (content != null) return JSON.stringify(content, null, 2);
  return "";
}

export function RequestMessagesView({ detail }: { detail: LiteLLMProxyRequestDetail }) {
  const [copied, setCopied] = useState(false);
  const messages = detail.messages;
  const fullText = promptFullText(detail);

  async function copyAll() {
    try {
      await navigator.clipboard.writeText(fullText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* ignore */
    }
  }

  if (!messages || messages.length === 0) {
    if (!fullText) {
      return <p className="text-sm text-gray-500">messages なし</p>;
    }
    return (
      <div className="space-y-2">
        <CopyBar copied={copied} onCopy={copyAll} />
        <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words font-mono max-h-96 overflow-y-auto rounded-lg border border-white/10 bg-bg-primary p-3">
          {fullText}
        </pre>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <CopyBar copied={copied} onCopy={copyAll} />
      {detail.messages_truncated && (
        <p className="text-xs text-amber-400">プロンプトはサイズ上限（64KB）のため一部のみ保存されています。</p>
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
        <div className="bg-bg-primary rounded px-2 py-1">
          <span className="text-gray-500">max_tokens </span>
          <span className="font-mono">{detail.max_tokens ?? "—"}</span>
        </div>
        <div className="bg-bg-primary rounded px-2 py-1">
          <span className="text-gray-500">messages </span>
          <span className="font-mono">{detail.message_count ?? messages.length}</span>
        </div>
        <div className="bg-bg-primary rounded px-2 py-1">
          <span className="text-gray-500">~chars </span>
          <span className="font-mono">{detail.prompt_char_est ?? "—"}</span>
        </div>
        <div className="bg-bg-primary rounded px-2 py-1">
          <span className="text-gray-500">model </span>
          <span className="font-mono truncate block">{detail.model}</span>
        </div>
      </div>
      {messages.map((msg: ChatMessage, i: number) => (
        <div key={i} className="rounded-lg border border-white/10 bg-bg-primary p-3">
          <p className="text-xs font-semibold text-accent-primary mb-1">{msg.role ?? "unknown"}</p>
          <pre className="text-xs text-gray-300 whitespace-pre-wrap break-words font-mono max-h-96 overflow-y-auto">
            {contentText(msg.content)}
          </pre>
        </div>
      ))}
    </div>
  );
}

function CopyBar({ copied, onCopy }: { copied: boolean; onCopy: () => void }) {
  return (
    <div className="flex justify-end">
      <button
        type="button"
        onClick={onCopy}
        className="inline-flex items-center gap-1 text-xs text-gray-400 hover:text-white px-2 py-1 rounded border border-white/10"
      >
        {copied ? <Check className="w-3 h-3 text-accent-success" /> : <Copy className="w-3 h-3" />}
        {copied ? "コピー済み" : "全文をコピー"}
      </button>
    </div>
  );
}
