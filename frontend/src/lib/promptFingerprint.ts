import type { ChatMessage, LiteLLMProxyRequestDetail } from "@/types";

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
  if (content != null) return JSON.stringify(content);
  return "";
}

/** グループ化・比較用の全文（messages 優先） */
export function promptFullText(row: Pick<LiteLLMProxyRequestDetail, "messages" | "request_summary">): string {
  const messages = row.messages;
  if (messages && messages.length > 0) {
    return messages
      .map((m: ChatMessage) => `[${m.role ?? "?"}]\n${contentText(m.content)}`)
      .join("\n\n");
  }
  return row.request_summary ?? "";
}

/** 先頭のみプレビュー */
export function promptPreview(row: Pick<LiteLLMProxyRequestDetail, "messages" | "request_summary">, max = 160): string {
  const full = promptFullText(row).replace(/\s+/g, " ").trim();
  if (!full) return "（本文なし）";
  return full.length <= max ? full : `${full.slice(0, max)}…`;
}

function hashString(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h << 5) - h + s.charCodeAt(i);
    h |= 0;
  }
  return String(h >>> 0);
}

/** 同一プロンプト判定用キー（空白正規化 + 先頭 8k 文字） */
export function promptGroupKey(
  row: Pick<LiteLLMProxyRequestDetail, "messages" | "request_summary" | "prompt_char_est" | "max_tokens">
): string {
  const norm = promptFullText(row).replace(/\s+/g, " ").trim().slice(0, 8000);
  return `${row.prompt_char_est ?? 0}|${row.max_tokens ?? ""}|${hashString(norm)}`;
}

export interface PromptGroup {
  key: string;
  count: number;
  rows: LiteLLMProxyRequestDetail[];
  preview: string;
  charEst: number;
  errorCount: number;
  firstAt: number;
  lastAt: number;
}

export function buildPromptGroups(rows: LiteLLMProxyRequestDetail[]): PromptGroup[] {
  const map = new Map<string, LiteLLMProxyRequestDetail[]>();
  for (const row of rows) {
    const key = promptGroupKey(row);
    const list = map.get(key) ?? [];
    list.push(row);
    map.set(key, list);
  }
  const groups: PromptGroup[] = [];
  for (const [key, items] of map) {
    const sorted = [...items].sort(
      (a, b) => (b.completed_at ?? b.started_at ?? 0) - (a.completed_at ?? a.started_at ?? 0)
    );
    const rep = sorted[0];
    const times = sorted.map((r) => r.completed_at ?? r.started_at ?? 0).filter(Boolean);
    groups.push({
      key,
      count: sorted.length,
      rows: sorted,
      preview: promptPreview(rep),
      charEst: rep.prompt_char_est ?? 0,
      errorCount: sorted.filter((r) => r.status === "error").length,
      firstAt: times.length ? Math.min(...times) : 0,
      lastAt: times.length ? Math.max(...times) : 0,
    });
  }
  return groups.sort((a, b) => b.count - a.count);
}
