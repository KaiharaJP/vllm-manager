"use client";

import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { LiteLLMProxyRequestDetail } from "@/types";
import { RequestMessagesView } from "@/components/RequestMessagesView";
import { buildPromptGroups, type PromptGroup } from "@/lib/promptFingerprint";
import { ChevronDown, ChevronRight, RefreshCw } from "lucide-react";

type ViewMode = "list" | "groups";

function formatTime(ts: number | undefined): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString();
}

function formatElapsed(seconds: number | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

export default function RequestHistoryPanel() {
  const [rows, setRows] = useState<LiteLLMProxyRequestDetail[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("groups");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [expandedGroupKey, setExpandedGroupKey] = useState<string | null>(null);
  const [detail, setDetail] = useState<LiteLLMProxyRequestDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const groups = useMemo(() => buildPromptGroups(rows), [rows]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getRequestHistory(200, 0);
      setRows(res.requests);
      setTotal(res.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function loadDetail(id: string) {
    setDetailLoading(true);
    try {
      const d = await api.getRequestHistoryDetail(id);
      setDetail(d);
    } catch (e) {
      setDetail({
        id,
        endpoint: "",
        model: "",
        stream: false,
        prompt_tokens: null,
        completion_tokens: null,
        total_tokens: null,
        completion_chunks: 0,
        status: "error",
        phase: "error",
        first_token_at: null,
        prefill_tok_s: null,
        gen_tok_s: null,
        elapsed_s: 0,
        error: e instanceof Error ? e.message : String(e),
        started_at: 0,
        updated_at: 0,
      });
    } finally {
      setDetailLoading(false);
    }
  }

  async function toggleExpandList(id: string) {
    if (expandedId === id) {
      setExpandedId(null);
      setDetail(null);
      return;
    }
    setExpandedGroupKey(null);
    setExpandedId(id);
    setDetail(null);
    await loadDetail(id);
  }

  async function toggleExpandGroup(group: PromptGroup) {
    if (expandedGroupKey === group.key) {
      setExpandedGroupKey(null);
      setExpandedId(null);
      setDetail(null);
      return;
    }
    setExpandedGroupKey(group.key);
    setExpandedId(null);
    setDetail(null);
    const rep = group.rows[0];
    if (rep.messages?.length) {
      setDetail(rep);
      return;
    }
    await loadDetail(rep.id);
  }

  return (
    <div className="space-y-4 animate-slide-in">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1 max-w-2xl">
          <h2 className="text-lg font-semibold">リクエスト履歴</h2>
          <p className="text-sm text-gray-400">
            LiteLLM 経由で記録されたリクエストのみ。vLLM の Waiting 60 本すべての中身は見えませんが、
            ここに溜まった件数・プロンプトの同一性は確認できます。
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg border border-white/10 overflow-hidden text-xs">
            <button
              type="button"
              onClick={() => setViewMode("groups")}
              className={`px-3 py-1.5 ${viewMode === "groups" ? "bg-white/10 text-white" : "text-gray-400"}`}
            >
              プロンプト別
            </button>
            <button
              type="button"
              onClick={() => setViewMode("list")}
              className={`px-3 py-1.5 ${viewMode === "list" ? "bg-white/10 text-white" : "text-gray-400"}`}
            >
              一覧
            </button>
          </div>
          <button
            type="button"
            onClick={load}
            className="inline-flex items-center gap-2 rounded-lg border border-white/15 px-3 py-2 text-sm text-gray-300 hover:bg-white/5"
          >
            <RefreshCw className="w-4 h-4" />
            更新
          </button>
        </div>
      </div>

      {!loading && rows.length > 0 && viewMode === "groups" && (
        <p className="text-xs text-gray-500 border border-white/10 rounded-lg px-3 py-2">
          表示 {rows.length} 件を <strong className="text-gray-300">{groups.length} 種類</strong> のプロンプトに集約。
          {groups[0] && groups[0].count > 1 && (
            <> 最多は <strong className="text-gray-300">{groups[0].count} 回</strong> 同じ内容（~{groups[0].charEst.toLocaleString()} 文字）。</>
          )}
        </p>
      )}

      {error && (
        <p className="text-sm text-accent-danger border border-accent-danger/30 rounded-lg px-3 py-2">{error}</p>
      )}

      {loading ? (
        <p className="text-gray-400 text-center py-8">読み込み中...</p>
      ) : rows.length === 0 ? (
        <p className="text-gray-400 border border-dashed border-white/20 rounded-lg p-6 text-center">
          履歴がありません。LiteLLM 経由で chat completions を実行するとここに記録されます。
        </p>
      ) : viewMode === "groups" ? (
        <GroupView
          groups={groups}
          expandedGroupKey={expandedGroupKey}
          detail={detail}
          detailLoading={detailLoading}
          onToggleGroup={toggleExpandGroup}
        />
      ) : (
        <ListView
          rows={rows}
          total={total}
          expandedId={expandedId}
          detail={detail}
          detailLoading={detailLoading}
          onToggle={toggleExpandList}
        />
      )}
    </div>
  );
}

function GroupView({
  groups,
  expandedGroupKey,
  detail,
  detailLoading,
  onToggleGroup,
}: {
  groups: PromptGroup[];
  expandedGroupKey: string | null;
  detail: LiteLLMProxyRequestDetail | null;
  detailLoading: boolean;
  onToggleGroup: (g: PromptGroup) => void;
}) {
  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-gray-500 border-b border-white/10">
              <th className="p-3 w-8" />
              <th className="p-3 text-right">件数</th>
              <th className="p-3 text-right">~chars</th>
              <th className="p-3 text-right">error</th>
              <th className="p-3">プロンプト先頭（クリックで全文）</th>
              <th className="p-3 whitespace-nowrap">期間</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <Fragment key={g.key}>
                <tr
                  className="border-b border-white/5 hover:bg-white/[0.02] cursor-pointer"
                  onClick={() => onToggleGroup(g)}
                >
                  <td className="p-3 text-gray-400">
                    {expandedGroupKey === g.key ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                  </td>
                  <td className="p-3 text-right font-mono font-semibold text-accent-warning">{g.count}</td>
                  <td className="p-3 text-right font-mono">{g.charEst.toLocaleString()}</td>
                  <td className="p-3 text-right font-mono text-accent-danger">{g.errorCount || "—"}</td>
                  <td className="p-3 text-xs text-gray-300 max-w-md">{g.preview}</td>
                  <td className="p-3 text-xs text-gray-500 whitespace-nowrap">
                    {formatTime(g.firstAt)} 〜 {formatTime(g.lastAt)}
                  </td>
                </tr>
                {expandedGroupKey === g.key && (
                  <tr className="bg-bg-tertiary/50">
                    <td colSpan={6} className="p-4 space-y-3">
                      <p className="text-xs text-gray-500">
                        同一プロンプト {g.count} 件（代表 1 件の全文。個別 ID は一覧タブで確認）
                      </p>
                      {detailLoading ? (
                        <p className="text-sm text-gray-400">読み込み中...</p>
                      ) : detail ? (
                        <RequestMessagesView detail={detail} />
                      ) : null}
                      <details className="text-xs">
                        <summary className="cursor-pointer text-gray-400 hover:text-white">
                          このグループのリクエスト ID（{g.count} 件）
                        </summary>
                        <ul className="mt-2 font-mono text-gray-500 space-y-0.5 max-h-32 overflow-y-auto">
                          {g.rows.map((r) => (
                            <li key={r.id}>
                              {r.id.slice(0, 8)}… {r.status} {formatElapsed(r.elapsed_s)}
                            </li>
                          ))}
                        </ul>
                      </details>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ListView({
  rows,
  total,
  expandedId,
  detail,
  detailLoading,
  onToggle,
}: {
  rows: LiteLLMProxyRequestDetail[];
  total: number;
  expandedId: string | null;
  detail: LiteLLMProxyRequestDetail | null;
  detailLoading: boolean;
  onToggle: (id: string) => void;
}) {
  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 overflow-hidden">
      <p className="text-xs text-gray-500 px-4 py-2 border-b border-white/5">
        全 {total} 件（表示 {rows.length} 件）— 行をクリックでプロンプト全文
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-gray-500 border-b border-white/10">
              <th className="p-3 w-8" />
              <th className="p-3">完了時刻</th>
              <th className="p-3">model</th>
              <th className="p-3">status</th>
              <th className="p-3 text-right">~chars</th>
              <th className="p-3 text-right">経過</th>
              <th className="p-3">サマリー</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <Fragment key={r.id}>
                <tr
                  className="border-b border-white/5 hover:bg-white/[0.02] cursor-pointer"
                  onClick={() => onToggle(r.id)}
                >
                  <td className="p-3 text-gray-400">
                    {expandedId === r.id ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                  </td>
                  <td className="p-3 font-mono text-xs whitespace-nowrap">
                    {formatTime(r.completed_at ?? r.updated_at)}
                  </td>
                  <td className="p-3 text-xs truncate max-w-[140px]">{r.model}</td>
                  <td className="p-3">
                    <span className={r.status === "error" ? "text-accent-danger" : "text-gray-400"}>
                      {r.status}
                    </span>
                  </td>
                  <td className="p-3 text-right font-mono">{r.prompt_char_est?.toLocaleString() ?? "—"}</td>
                  <td className="p-3 text-right font-mono">{formatElapsed(r.elapsed_s)}</td>
                  <td className="p-3 text-xs text-gray-400 max-w-[200px] truncate">
                    {r.request_summary ?? "—"}
                  </td>
                </tr>
                {expandedId === r.id && (
                  <tr className="bg-bg-tertiary/50">
                    <td colSpan={7} className="p-4">
                      {detailLoading ? (
                        <p className="text-sm text-gray-400">読み込み中...</p>
                      ) : detail && detail.id === r.id ? (
                        <RequestMessagesView detail={detail} />
                      ) : null}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
