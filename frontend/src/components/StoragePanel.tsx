"use client";

import { useEffect, useState } from "react";
import { HardDrive, FolderOpen, RefreshCw, ChevronRight, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import type {
  StorageBreakdown,
  StorageMount,
  StorageUsageReport,
} from "@/types";

function formatGb(gb: number | null | undefined): string {
  if (gb == null) return "—";
  if (gb >= 1024) return `${(gb / 1024).toFixed(2)} TB`;
  return `${gb.toFixed(gb >= 100 ? 0 : 1)} GB`;
}

function barColor(percent: number): string {
  if (percent >= 90) return "bg-accent-danger";
  if (percent >= 75) return "bg-accent-warning";
  return "bg-accent-primary";
}

export default function StoragePanel() {
  const [mounts, setMounts] = useState<StorageMount[]>([]);
  const [mountsError, setMountsError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getStorageOverview()
      .then(setMounts)
      .catch((e) => setMountsError(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="space-y-6 animate-slide-in">
      {/* ドライブ概要 */}
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <HardDrive className="w-5 h-5 text-accent-primary" />
          ドライブ使用状況
        </h2>
        {mountsError && <p className="text-sm text-accent-warning">{mountsError}</p>}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {mounts
            .filter((m) => !m.same_device_as_above)
            .map((m) => {
              const percent = m.used_percent ?? 0;
              return (
                <div key={m.path} className="bg-bg-primary rounded-lg p-4">
                  <div className="flex justify-between items-baseline mb-1">
                    <span className="text-sm font-medium">{m.label}</span>
                    <span className="text-xs text-gray-500">{m.path}</span>
                  </div>
                  <div className="h-2.5 bg-white/5 rounded-full overflow-hidden mb-2">
                    <div
                      className={`h-full rounded-full ${barColor(percent)}`}
                      style={{ width: `${Math.min(100, percent)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-400">
                    <span>
                      {formatGb(m.used_gb)} / {formatGb(m.total_gb)}（{percent.toFixed(1)}%）
                    </span>
                    <span className={m.free_gb < 150 ? "text-accent-danger font-medium" : ""}>
                      空き {formatGb(m.free_gb)}
                    </span>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      <DirectoryExplorer mounts={mounts} />
      <UsageReport />
    </div>
  );
}

/** 探索対象として選べるドライブ（重複デバイスは除外） */
function driveChoices(mounts: StorageMount[]): StorageMount[] {
  return mounts.filter((m) => !m.same_device_as_above);
}

/** ディレクトリを掘って何が容量を食っているか調べるエクスプローラ */
function DirectoryExplorer({ mounts }: { mounts: StorageMount[] }) {
  const [drive, setDrive] = useState<StorageMount | null>(null);
  const [path, setPath] = useState<string>("");
  const [data, setData] = useState<StorageBreakdown | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load(target: string, refresh = false) {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getStorageBreakdown(target, refresh);
      setData(result);
      setPath(result.path);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function selectDrive(m: StorageMount) {
    setDrive(m);
    setData(null);
    void load(m.path);
  }

  function backToDrives() {
    setDrive(null);
    setData(null);
    setError(null);
    setPath("");
  }

  // パンくず: ドライブのルートから現在パスまでのセグメント
  const crumbs: { label: string; target: string }[] = [];
  if (drive) {
    crumbs.push({ label: drive.label, target: drive.path });
    const rootPath = drive.path === "/" ? "" : drive.path;
    if (path.startsWith(rootPath) && path !== drive.path) {
      const rest = path.slice(rootPath.length).replace(/^\//, "");
      let acc = rootPath;
      for (const seg of rest.split("/")) {
        if (!seg) continue;
        acc = `${acc}/${seg}`;
        crumbs.push({ label: seg, target: acc });
      }
    }
  }

  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
      <h2 className="text-lg font-semibold mb-1 flex items-center gap-2">
        <FolderOpen className="w-5 h-5 text-accent-primary" />
        ディレクトリ内訳
      </h2>

      {/* Step 1: ドライブ選択 */}
      {!drive && (
        <>
          <p className="text-sm text-gray-400 mb-4">
            調べたいドライブを選んでください。フォルダをクリックすると中を掘っていけます。
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {driveChoices(mounts).map((m) => {
              const percent = m.used_percent ?? 0;
              return (
                <button
                  key={m.path}
                  onClick={() => selectDrive(m)}
                  className="text-left bg-bg-primary rounded-xl border border-white/10 p-5 hover:border-accent-primary/60 hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="flex items-center gap-2 font-medium">
                      <HardDrive className="w-5 h-5 text-accent-primary" />
                      {m.label}
                    </span>
                    <ChevronRight className="w-5 h-5 text-gray-500" />
                  </div>
                  <div className="h-2 bg-white/5 rounded-full overflow-hidden mb-2">
                    <div
                      className={`h-full rounded-full ${barColor(percent)}`}
                      style={{ width: `${Math.min(100, percent)}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-400">
                    {formatGb(m.used_gb)} 使用 / {formatGb(m.total_gb)}（空き {formatGb(m.free_gb)}）
                  </p>
                  <p className="text-xs text-gray-500 font-mono mt-1">{m.path}</p>
                </button>
              );
            })}
            {mounts.length === 0 && (
              <p className="text-sm text-gray-500">ドライブ情報を読み込み中...</p>
            )}
          </div>
        </>
      )}

      {/* Step 2: ドリルダウン */}
      {drive && (
        <>
          <div className="flex items-center justify-between gap-2 mb-4 flex-wrap">
            {/* パンくず */}
            <div className="flex items-center gap-1 text-sm flex-wrap min-w-0">
              <button
                onClick={backToDrives}
                className="px-2 py-1 rounded-md bg-white/5 text-gray-400 hover:text-white text-xs shrink-0"
              >
                ← ドライブ選択
              </button>
              {crumbs.map((c, i) => (
                <span key={c.target} className="flex items-center gap-1 min-w-0">
                  {i > 0 && <ChevronRight className="w-3.5 h-3.5 text-gray-600 shrink-0" />}
                  <button
                    onClick={() => void load(c.target)}
                    disabled={loading || i === crumbs.length - 1}
                    className={`truncate max-w-[180px] ${
                      i === crumbs.length - 1
                        ? "text-white font-medium"
                        : "text-accent-primary hover:underline"
                    }`}
                  >
                    {c.label}
                  </button>
                </span>
              ))}
            </div>
            <button
              onClick={() => void load(path, true)}
              disabled={loading}
              className="p-1.5 rounded-md bg-white/5 text-gray-400 hover:text-white disabled:opacity-50 shrink-0"
              title="再スキャン（大きなディレクトリは数分かかります）"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>

          {/* 現在地のサマリ */}
          {data?.total_gb != null && (
            <p className="text-sm text-gray-400 mb-3">
              <span className="font-mono text-gray-300">{path}</span> の合計{" "}
              <span className="text-white font-medium">{formatGb(data.total_gb)}</span>
              {data.cached && (
                <span className="text-xs text-gray-500 ml-2">
                  （{Math.round((data.age_sec ?? 0) / 60)} 分前のスキャン結果）
                </span>
              )}
            </p>
          )}

          {error && <p className="text-sm text-accent-warning mb-3">{error}</p>}
          {loading && (
            <p className="text-sm text-gray-400 flex items-center gap-2 mb-3">
              <Loader2 className="w-4 h-4 animate-spin" />
              スキャン中...（初回や再スキャンは数分かかることがあります）
            </p>
          )}

          <div className="space-y-1">
            {data?.entries.map((e) => {
              const name = e.path.split("/").pop() || e.path;
              const total = data.total_gb || 0;
              const share = total > 0 ? (e.size_gb / total) * 100 : 0;
              return (
                <button
                  key={e.path}
                  onClick={() => void load(e.path)}
                  disabled={loading}
                  className="w-full text-left group relative rounded-md overflow-hidden bg-bg-primary hover:bg-white/5 transition-colors"
                >
                  <div
                    className="absolute inset-y-0 left-0 bg-accent-primary/10 group-hover:bg-accent-primary/20"
                    style={{ width: `${Math.max(1, share)}%` }}
                  />
                  <div className="relative flex items-center gap-3 px-3 py-2 text-sm">
                    <FolderOpen className="w-4 h-4 text-accent-primary/70 shrink-0" />
                    <span className="font-mono truncate flex-1">{name}</span>
                    <span className="text-gray-500 tabular-nums text-xs shrink-0 w-14 text-right">
                      {share >= 0.1 ? `${share.toFixed(1)}%` : "<0.1%"}
                    </span>
                    <span className="text-gray-300 tabular-nums shrink-0 w-24 text-right">
                      {formatGb(e.size_gb)}
                    </span>
                    <ChevronRight className="w-4 h-4 text-gray-600 group-hover:text-gray-400 shrink-0" />
                  </div>
                </button>
              );
            })}
            {data && data.entries.length === 0 && !loading && (
              <p className="text-sm text-gray-500">サブディレクトリはありません</p>
            )}
          </div>
          {data && data.entries_omitted > 0 && (
            <p className="text-xs text-gray-500 mt-2">
              …他 {data.entries_omitted} 件（小さい順に省略）
            </p>
          )}
          {data && data.warnings.length > 0 && (
            <p className="text-[11px] text-gray-500 mt-2">
              一部読み取れないディレクトリがあり、サイズが過小の可能性があります（権限制限）
            </p>
          )}
        </>
      )}
    </div>
  );
}

/** 用途別（HF キャッシュのモデル別 / Ollama / 学習ジョブ）内訳 */
function UsageReport() {
  const [report, setReport] = useState<StorageUsageReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load(refresh = false) {
    setLoading(true);
    setError(null);
    try {
      setReport(await api.getStorageUsage(refresh));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  return (
    <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold">用途別内訳（モデル・ジョブ）</h2>
        <button
          onClick={() => void load(true)}
          disabled={loading}
          className="p-1.5 rounded-md bg-white/5 text-gray-400 hover:text-white disabled:opacity-50"
          title="再スキャン（数分かかります）"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>
      <p className="text-xs text-gray-500 mb-4">
        HuggingFace キャッシュはモデル別、Ollama はモデル:タグ別、学習ジョブはジョブ別のサイズ。
        {report?.cached && ` キャッシュ済みの結果（${Math.round((report.age_sec ?? 0) / 60)} 分前）。`}
      </p>
      {error && <p className="text-sm text-accent-warning mb-3">{error}</p>}
      {loading && !report && (
        <p className="text-sm text-gray-400 flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin" />
          スキャン中...
        </p>
      )}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {report?.sections.map((s) => {
          const max = s.items[0]?.size_gb || 1;
          return (
            <div key={s.category} className="bg-bg-primary rounded-lg p-4">
              <div className="flex justify-between items-baseline mb-1">
                <h3 className="text-sm font-medium">{s.category}</h3>
                <span className="text-xs text-gray-400">{formatGb(s.total_gb)}</span>
              </div>
              <p className="text-xs text-gray-500 font-mono mb-2">{s.path}</p>
              <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
                {s.items.map((i) => (
                  <div key={i.name} className="relative rounded overflow-hidden">
                    <div
                      className="absolute inset-y-0 left-0 bg-accent-primary/10"
                      style={{ width: `${Math.max(1, (i.size_gb / max) * 100)}%` }}
                    />
                    <div className="relative flex justify-between px-2 py-1 text-xs">
                      <span className="truncate">{i.name}</span>
                      <span className="text-gray-400 tabular-nums shrink-0 ml-2">
                        {formatGb(i.size_gb)}
                      </span>
                    </div>
                  </div>
                ))}
                {s.items.length === 0 && (
                  <p className="text-xs text-gray-500">なし</p>
                )}
              </div>
              {s.note && <p className="text-[11px] text-gray-500 mt-2">{s.note}</p>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
