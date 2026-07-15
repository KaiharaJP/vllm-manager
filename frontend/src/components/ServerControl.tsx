"use client";

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api } from "@/lib/api";
import type {
  ServerStatus,
  ServerConfig,
  Model,
  ContextPreset,
  ServerStartRequest,
  ApiResponse,
  RunningServer,
  SystemGpuMetrics,
  SmokeTestResult,
} from "@/types";
import { Copy, Info, Play, StopCircle, RotateCcw, Terminal } from "lucide-react";

interface ServerControlProps {
  status: ServerStatus | null;
  config: ServerConfig | null;
  models: Model[];
  contextPresets: ContextPreset[];
  onActionComplete: () => void;
}

type RejectionSampleMethod = "standard" | "synthetic";
type SpecMethod =
  | "mtp"
  | "qwen3_next_mtp"
  | "qwen3_5_mtp"
  | "eagle3"
  | "draft_model"
  | "ngram"
  | "suffix";

type SpecForm = {
  method: SpecMethod;
  num_speculative_tokens: number;
  model: string;
  draft_tensor_parallel_size: number;
  max_model_len: number;
  parallel_drafting: boolean;
  rejection_sample_method: RejectionSampleMethod;
  synthetic_acceptance_rate: number;
  prompt_lookup_min: number;
  prompt_lookup_max: number;
  suffix_decoding_max_tree_depth: number;
  suffix_decoding_max_cached_requests: number;
  suffix_decoding_max_spec_factor: number;
  suffix_decoding_min_token_prob: number;
};

function asNumber(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function asBool(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value;
  return fallback;
}

function modelLikelySupportsVision(modelId: string): boolean {
  const lowered = modelId.toLowerCase();
  const hints = [
    "qwen3-vl",
    "qwen2-vl",
    "qwen3.6",
    "qwen3.5",
    "qwen2.5-vl",
    "llava",
    "phi-3.5-vision",
    "internvl",
    "minicpm-v",
  ];
  return hints.some((h) => lowered.includes(h));
}

function limitMmImageFromConfig(
  raw: Record<string, number> | null | undefined
): number {
  if (!raw || typeof raw.image !== "number") return 1;
  return Math.max(0, Math.min(8, Math.floor(raw.image)));
}

function createSpecForm(raw: Record<string, unknown> | null | undefined): SpecForm {
  const data = raw ?? {};
  const methodRaw = typeof data.method === "string" ? data.method : "ngram";
  const method: SpecMethod = (
    [
      "mtp",
      "qwen3_next_mtp",
      "qwen3_5_mtp",
      "eagle3",
      "draft_model",
      "ngram",
      "suffix",
    ] as const
  ).includes(methodRaw as SpecMethod)
    ? (methodRaw as SpecMethod)
    : "ngram";
  const rejectMethodRaw =
    typeof data.rejection_sample_method === "string" ? data.rejection_sample_method : "standard";
  const rejectionSampleMethod: RejectionSampleMethod =
    rejectMethodRaw === "synthetic"
      ? "synthetic"
      : "standard";
  return {
    method,
    num_speculative_tokens: Math.max(1, Math.floor(asNumber(data.num_speculative_tokens, 2))),
    model: typeof data.model === "string" ? data.model : "",
    draft_tensor_parallel_size: Math.max(1, Math.floor(asNumber(data.draft_tensor_parallel_size, 1))),
    max_model_len: Math.max(1, Math.floor(asNumber(data.max_model_len, 8192))),
    parallel_drafting: asBool(data.parallel_drafting, false),
    rejection_sample_method: rejectionSampleMethod,
    synthetic_acceptance_rate: Math.max(0, Math.min(1, asNumber(data.synthetic_acceptance_rate, 0.7))),
    prompt_lookup_min: Math.max(1, Math.floor(asNumber(data.prompt_lookup_min, 2))),
    prompt_lookup_max: Math.max(1, Math.floor(asNumber(data.prompt_lookup_max, 5))),
    suffix_decoding_max_tree_depth: Math.max(
      1,
      Math.floor(asNumber(data.suffix_decoding_max_tree_depth, 24))
    ),
    suffix_decoding_max_cached_requests: Math.max(
      0,
      Math.floor(asNumber(data.suffix_decoding_max_cached_requests, 10000))
    ),
    suffix_decoding_max_spec_factor: Math.max(
      0.1,
      asNumber(data.suffix_decoding_max_spec_factor, 1.0)
    ),
    suffix_decoding_min_token_prob: Math.max(
      0,
      Math.min(1, asNumber(data.suffix_decoding_min_token_prob, 0.1))
    ),
  };
}

function buildSpeculativeConfig(enabled: boolean, spec: SpecForm): Record<string, unknown> {
  if (!enabled) return {};
  const cfg: Record<string, unknown> = {
    method: spec.method,
    num_speculative_tokens: Math.max(1, Math.floor(spec.num_speculative_tokens)),
    rejection_sample_method: spec.rejection_sample_method,
  };
  if (spec.model.trim()) cfg.model = spec.model.trim();
  if (spec.draft_tensor_parallel_size > 0) {
    cfg.draft_tensor_parallel_size = Math.floor(spec.draft_tensor_parallel_size);
  }
  if (spec.max_model_len > 0) cfg.max_model_len = Math.floor(spec.max_model_len);
  if (spec.parallel_drafting) cfg.parallel_drafting = true;
  if (spec.rejection_sample_method === "synthetic") {
    cfg.synthetic_acceptance_rate = Math.max(0, Math.min(1, spec.synthetic_acceptance_rate));
  }
  if (spec.method === "ngram") {
    cfg.prompt_lookup_min = Math.max(1, Math.floor(spec.prompt_lookup_min));
    cfg.prompt_lookup_max = Math.max(1, Math.floor(spec.prompt_lookup_max));
  }
  if (spec.method === "suffix") {
    cfg.suffix_decoding_max_tree_depth = Math.max(
      1,
      Math.floor(spec.suffix_decoding_max_tree_depth)
    );
    cfg.suffix_decoding_max_cached_requests = Math.max(
      0,
      Math.floor(spec.suffix_decoding_max_cached_requests)
    );
    cfg.suffix_decoding_max_spec_factor = Math.max(0.1, spec.suffix_decoding_max_spec_factor);
    cfg.suffix_decoding_min_token_prob = Math.max(
      0,
      Math.min(1, spec.suffix_decoding_min_token_prob)
    );
  }
  return cfg;
}

const INPUT_CLASS =
  "w-full max-w-xs bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white tabular-nums focus:outline-none focus:ring-2 focus:ring-accent-primary disabled:opacity-50";
const SLIDER_CLASS = "w-full accent-accent-primary disabled:opacity-50";

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function parseNumberInput(raw: string, integer: boolean): number | null {
  if (raw.trim() === "") return null;
  const parsed = integer ? parseInt(raw, 10) : parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

const LOG_SLIDER_STEPS = 1000;

function snapToStep(value: number, step: number): number {
  if (step <= 0) return value;
  return Math.round(value / step) * step;
}

function valueToLogSliderPos(value: number, sMin: number, sMax: number): number {
  const v = clampNumber(value, sMin, sMax);
  if (sMin <= 0 || sMax <= sMin) return 0;
  const logMin = Math.log(sMin);
  const logMax = Math.log(sMax);
  return Math.round(((Math.log(v) - logMin) / (logMax - logMin)) * LOG_SLIDER_STEPS);
}

function logSliderPosToValue(pos: number, sMin: number, sMax: number, step: number): number {
  const t = clampNumber(pos, 0, LOG_SLIDER_STEPS) / LOG_SLIDER_STEPS;
  const logMin = Math.log(sMin);
  const logMax = Math.log(sMax);
  const raw = Math.exp(logMin + t * (logMax - logMin));
  return clampNumber(snapToStep(raw, step), sMin, sMax);
}

type NumberPreset = {
  label: string;
  value: number;
  hint?: string;
};

/** デフォルト max_tokens のプリセット（起動フォーム・選択用） */
const DEFAULT_MAX_TOKENS_PRESETS: NumberPreset[] = [
  { label: "短文", value: 512, hint: "タイトル生成・要約" },
  { label: "標準", value: 2048, hint: "通常のチャット" },
  { label: "エージェント", value: 4096, hint: "Hermes 等（推奨）" },
  { label: "長文", value: 8192, hint: "レポート・長いコード" },
  { label: "特大", value: 16384, hint: "キュー占有に注意" },
];

const DEFAULT_MAX_TOKENS_FALLBACK = 4096;

function copyTextWithFallback(text: string): boolean {
  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    textarea.style.pointerEvents = "none";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
}

export default function ServerControl({
  status,
  config,
  models,
  contextPresets,
  onActionComplete,
}: ServerControlProps) {
  const downloadedModels = models.filter((m) => Boolean(m.downloaded));
  const [form, setForm] = useState({
    model_id: config?.model_id ?? downloadedModels[0]?.id ?? "",
    instance_name: "",
    context_length: config?.context_length ?? 131072,
    max_num_seqs: config?.max_num_seqs ?? 6,
    default_max_tokens: config?.default_max_tokens ?? DEFAULT_MAX_TOKENS_FALLBACK,
    default_temperature: config?.default_temperature ?? 0.7,
    default_top_p: config?.default_top_p ?? 0.95,
    default_frequency_penalty: config?.default_frequency_penalty ?? 0,
    default_presence_penalty: config?.default_presence_penalty ?? 0,
    gpu_memory_mode: config?.gpu_memory_mode ?? "auto",
    gpu_memory_utilization: config?.gpu_memory_utilization ?? 0.85,
    tensor_parallel_size: config?.tensor_parallel_size ?? 1,
    gpu_devices: config?.gpu_devices ?? "all",
    download_model: true,
    enable_auto_tool_choice: config?.enable_auto_tool_choice ?? false,
    tool_call_parser: config?.tool_call_parser ?? "",
    force_stream: config?.force_stream ?? true,
    limit_mm_image: limitMmImageFromConfig(config?.limit_mm_per_prompt),
    mm_encoder_tp_mode: config?.mm_encoder_tp_mode ?? "",
    mm_processor_cache_type: config?.mm_processor_cache_type ?? "",
  });
  const [specEnabled, setSpecEnabled] = useState(
    Boolean(config?.speculative_config && Object.keys(config.speculative_config).length > 0)
  );
  const [specForm, setSpecForm] = useState(
    createSpecForm((config?.speculative_config as Record<string, unknown> | null | undefined) ?? null)
  );

  const [action, setAction] = useState<"idle" | "starting" | "stopping" | "restarting">("idle");
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [showSteps, setShowSteps] = useState(false);
  const [gpuOptions, setGpuOptions] = useState<Array<{ index: number; name: string }>>([]);
  const [gpuMetrics, setGpuMetrics] = useState<SystemGpuMetrics[]>([]);
  const [runningServers, setRunningServers] = useState<RunningServer[]>([]);
  const [stoppingPid, setStoppingPid] = useState<number | null>(null);
  const [stoppingInstanceId, setStoppingInstanceId] = useState<string | null>(null);
  const [copyMessage, setCopyMessage] = useState<string | null>(null);
  const [smokeTestingId, setSmokeTestingId] = useState<string | null>(null);
  const [smokeTestResults, setSmokeTestResults] = useState<Record<string, SmokeTestResult>>({});

  useEffect(() => {
    const loadGpuOptions = async () => {
      try {
        const metrics = await api.getSystemMetrics();
        setGpuOptions(metrics.gpus.map((gpu) => ({ index: gpu.index, name: gpu.name })));
        setGpuMetrics(metrics.gpus);
      } catch {
        setGpuOptions([]);
        setGpuMetrics([]);
      }
    };
    loadGpuOptions();
  }, []);

  useEffect(() => {
    const loadRunningServers = async () => {
      try {
        const data = await api.getRunningServers();
        setRunningServers(data);
      } catch {
        setRunningServers([]);
      }
    };
    loadRunningServers();
    const timer = setInterval(loadRunningServers, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!downloadedModels.length) return;
    const exists = downloadedModels.some((m) => m.id === form.model_id);
    if (!exists) {
      setForm((prev) => ({ ...prev, model_id: downloadedModels[0].id }));
    }
  }, [downloadedModels, form.model_id]);

  useEffect(() => {
    const model = downloadedModels.find((m) => m.id === form.model_id);
    const recommendedContext = model?.recommended_context_length;
    if (!recommendedContext) return;
    setForm((prev) => {
      if (prev.model_id !== model.id) return prev;
      if (prev.context_length === recommendedContext) return prev;
      return { ...prev, context_length: recommendedContext };
    });
  }, [downloadedModels, form.model_id]);

  useEffect(() => {
    if (!config) return;
    setForm((prev) => ({
      ...prev,
      model_id: config.model_id ?? prev.model_id,
      context_length: config.context_length ?? prev.context_length,
      max_num_seqs: config.max_num_seqs ?? prev.max_num_seqs,
      default_max_tokens: config.default_max_tokens ?? prev.default_max_tokens,
      default_temperature: config.default_temperature ?? prev.default_temperature,
      default_top_p: config.default_top_p ?? prev.default_top_p,
      default_frequency_penalty: config.default_frequency_penalty ?? prev.default_frequency_penalty,
      default_presence_penalty: config.default_presence_penalty ?? prev.default_presence_penalty,
      gpu_memory_mode: config.gpu_memory_mode ?? prev.gpu_memory_mode,
      gpu_memory_utilization: config.gpu_memory_utilization ?? prev.gpu_memory_utilization,
      tensor_parallel_size: config.tensor_parallel_size ?? prev.tensor_parallel_size,
      gpu_devices: config.gpu_devices ?? prev.gpu_devices,
      enable_auto_tool_choice: config.enable_auto_tool_choice ?? prev.enable_auto_tool_choice,
      tool_call_parser: config.tool_call_parser ?? prev.tool_call_parser,
      force_stream: config.force_stream ?? prev.force_stream,
      limit_mm_image: limitMmImageFromConfig(config.limit_mm_per_prompt ?? undefined),
      mm_encoder_tp_mode: config.mm_encoder_tp_mode ?? prev.mm_encoder_tp_mode,
      mm_processor_cache_type:
        config.mm_processor_cache_type ?? prev.mm_processor_cache_type,
    }));
  }, [config]);

  async function handleStart() {
    setAction("starting");
    setResult(null);
    setShowSteps(true);
    try {
      const {
        limit_mm_image,
        mm_encoder_tp_mode,
        mm_processor_cache_type,
        ...startFields
      } = form;
      const payload: ServerStartRequest = {
        ...startFields,
        speculative_config: isPoolingTask ? {} : buildSpeculativeConfig(specEnabled, specForm),
        limit_mm_per_prompt: isPoolingTask ? undefined : { image: limit_mm_image },
        mm_encoder_tp_mode: isPoolingTask ? undefined : mm_encoder_tp_mode || undefined,
        mm_processor_cache_type: isPoolingTask ? undefined : mm_processor_cache_type || undefined,
        task_type: selectedTaskType,
        trust_remote_code: Boolean(selectedModel?.trust_remote_code),
        instance_name: form.instance_name.trim() || undefined,
        create_new_instance: true,
      };
      const res = await api.startServer(payload);
      setResult(res);
      onActionComplete();
    } catch (err) {
      setResult({ success: false, message: String(err) });
    } finally {
      setAction("idle");
    }
  }

  async function handleStop() {
    setAction("stopping");
    try {
      const res = await api.stopServer();
      setResult(res);
      onActionComplete();
    } catch (err) {
      setResult({ success: false, message: String(err) });
    } finally {
      setAction("idle");
    }
  }

  async function handleRestart() {
    setAction("restarting");
    setResult(null);
    setShowSteps(true);
    try {
      const res = await api.restartServer();
      setResult(res);
      onActionComplete();
    } catch (err) {
      setResult({ success: false, message: String(err) });
    } finally {
      setAction("idle");
    }
  }

  async function handleStopByInstance(instanceId: string) {
    setStoppingInstanceId(instanceId);
    try {
      const res = await api.stopInstance(instanceId);
      setResult({ success: res.success, message: res.message });
      onActionComplete();
      const data = await api.getRunningServers();
      setRunningServers(data);
    } catch (err) {
      setResult({ success: false, message: String(err) });
    } finally {
      setStoppingInstanceId(null);
    }
  }

  async function handleSmokeTest(instanceId: string) {
    setSmokeTestingId(instanceId);
    try {
      const res = await api.runSmokeTest(instanceId);
      setSmokeTestResults((prev) => ({ ...prev, [instanceId]: res }));
    } catch (err) {
      setSmokeTestResults((prev) => ({
        ...prev,
        [instanceId]: {
          instance_id: instanceId,
          success: false,
          task_type: null,
          latency_ms: null,
          tokens_generated: null,
          tokens_per_sec: null,
          response_preview: null,
          error: String(err),
        },
      }));
    } finally {
      setSmokeTestingId(null);
    }
  }

  async function handleStopByPid(pid: number) {
    setStoppingPid(pid);
    try {
      const res = await api.stopServerByPid(pid);
      setResult({ success: res.success, message: res.message });
      onActionComplete();
      const data = await api.getRunningServers();
      setRunningServers(data);
    } catch (err) {
      setResult({ success: false, message: String(err) });
    } finally {
      setStoppingPid(null);
    }
  }

  async function handleCopyAllServerCommands() {
    if (!runningServers.length) {
      setCopyMessage("コピー対象のサーバーがありません。");
      return;
    }
    const commands = runningServers
      .map((server) => server.command?.trim())
      .filter((cmd): cmd is string => Boolean(cmd));
    if (!commands.length) {
      setCopyMessage("起動コマンドを取得できませんでした。");
      return;
    }
    const text = commands.join("\n\n");
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else if (!copyTextWithFallback(text)) {
        throw new Error("Clipboard API unavailable");
      }
      setCopyMessage(`${commands.length} 件の起動コマンドをコピーしました。`);
    } catch {
      if (copyTextWithFallback(text)) {
        setCopyMessage(`${commands.length} 件の起動コマンドをコピーしました。`);
        return;
      }
      setCopyMessage(
        "コピーに失敗しました（HTTP配信やブラウザ権限でブロックされる場合があります）。"
      );
    }
  }

  const isBusy = action !== "idle";
  const hasDownloadedModels = downloadedModels.length > 0;
  const selectedModel = downloadedModels.find((m) => m.id === form.model_id);
  const selectedTaskType = selectedModel?.task_type ?? "chat";
  const isPoolingTask = selectedTaskType === "embedding" || selectedTaskType === "rerank";
  const taskTypeLabel =
    selectedTaskType === "embedding"
      ? "embedding"
      : selectedTaskType === "rerank"
        ? "rerank"
        : "LLM";

  return (
    <div className="space-y-6 animate-slide-in">
      {/* 操作パネル */}
      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Terminal className="w-5 h-5 text-accent-primary" />
          サーバー操作
          <InfoTooltip text="選択したモデルと推論パラメータで vLLM サーバーを起動・停止します。初回起動時はモデルのダウンロードに時間がかかります。" />
        </h2>

        {/* モデル選択 */}
        <div className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
          <FieldLabel
            label="モデル"
            hint="Hugging Face のモデル ID（例: org/name）です。vLLM 起動時はこの ID が `vllm serve <model>` に渡ります。一覧にはダウンロード済みモデルのみ出ます。未登録・未DL は「モデル管理」で追加・取得してください。量子化形式によっては vLLM 非対応のものがあります。"
          />
          <select
            value={form.model_id}
            onChange={(e) => setForm({ ...form, model_id: e.target.value })}
            disabled={isBusy || !hasDownloadedModels}
            className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-primary disabled:opacity-50"
          >
            {downloadedModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.size}) [
                {m.task_type === "embedding"
                  ? "embedding"
                  : m.task_type === "rerank"
                    ? "rerank"
                    : "LLM"}
                ]
              </option>
            ))}
          </select>
          {!hasDownloadedModels && (
            <p className="mt-1 text-xs text-gray-500">
              ダウンロード済みモデルがありません。先に「モデル管理」でモデルをダウンロードしてください。
            </p>
          )}
          </div>
          <div>
            <FieldLabel
              label="インスタンス名（任意）"
              hint="複数モデルを同時起動するときの表示名です。空欄なら model ID から自動生成されます。既存インスタンスは停止せず、新しい vLLM プロセスが追加されます。"
            />
            <input
              value={form.instance_name}
              onChange={(e) => setForm({ ...form, instance_name: e.target.value })}
              disabled={isBusy}
              placeholder="例: chat-main / embed-jina"
              className="w-full bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-primary disabled:opacity-50"
            />
          </div>
        </div>
        {isPoolingTask && (
          <div className="mb-4 space-y-1 text-xs text-accent-primary">
            <p>
              {selectedTaskType === "rerank" ? "Reranker" : "埋め込み"}モデルは vLLM を `--runner pooling` で起動します。LLM
              向けオプション（Speculative / Tool calling / Vision）は省略されます。
            </p>
            {selectedModel?.recommended_context_length && (
              <p>
                推奨コンテキスト長: {selectedModel.recommended_context_length.toLocaleString()} トークン
                （モデル選択時に起動フォームへ反映されます）
              </p>
            )}
            {selectedTaskType === "embedding" && selectedModel?.output_dimension && (
              <p>出力次元: {selectedModel.output_dimension}（smoke test で確認できます）</p>
            )}
            {selectedModel?.license_note && (
              <p className="text-amber-400">ライセンス: {selectedModel.license_note}</p>
            )}
            <p className="text-amber-400">
              大規模 LLM と同じ GPU に常駐させると VRAM 不足になることがあります。必要なら
              LLM を停止してから {taskTypeLabel} を起動してください。
            </p>
          </div>
        )}

        {/* コンテキスト長 */}
        <div className="mb-4">
          <FieldLabel
            label="コンテキスト長"
            hint="vLLM の `--max-model-len` に対応します。入力＋生成で扱えるトークン上限の目安です。大きくすると KV キャッシュ用 VRAM が増え、同じ GPU でも取れる `max_tokens` や同時リクエスト数が減りやすくなります。256K は非常に重いので、まず 32K〜64K から試すのが安全です。"
          />
          <div className="mb-2">
            <NumberSliderField
              label="トークン数"
              hint="プリセット以外の値も直接入力できます（4096〜262144）。"
              value={form.context_length}
              onChange={(context_length) => setForm({ ...form, context_length })}
              min={4096}
              max={262144}
              step={1024}
              disabled={isBusy}
            />
          </div>
          <div className="flex gap-2 flex-wrap">
            {contextPresets.map((p) => (
              <button
                key={p.value}
                type="button"
                onClick={() => setForm({ ...form, context_length: p.value })}
                disabled={isBusy}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  form.context_length === p.value
                    ? "bg-accent-primary text-white"
                    : "bg-bg-tertiary text-gray-400 hover:text-white border border-white/10"
                } disabled:opacity-50`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* スロット数 */}
        {!isPoolingTask && (
        <NumberSliderField
          className="mb-4"
          label="最大同時リクエスト数（max_num_seqs）"
          hint="vLLM の max-num-seqs。Hermes など並列リクエストが多い場合は 6〜8 程度を検討（1〜20）。"
          value={form.max_num_seqs}
          onChange={(max_num_seqs) => setForm({ ...form, max_num_seqs })}
          min={1}
          max={20}
          step={1}
          disabled={isBusy}
        />
        )}

        {/* デフォルト max_tokens */}
        {!isPoolingTask && (
        <div className="mb-4">
          <NumberSliderField
            label="デフォルト max_tokens"
            hint="クライアントが max_tokens を省略したときにプロキシが注入する上限です。プリセットを選ぶか、数値入力・スライダーで調整できます。"
            value={form.default_max_tokens}
            onChange={(default_max_tokens) => setForm({ ...form, default_max_tokens })}
            min={1}
            max={262144}
            step={1}
            sliderMin={256}
            sliderMax={131072}
            sliderStep={256}
            sliderScale="log"
            sliderHint="対数スケール"
            presets={DEFAULT_MAX_TOKENS_PRESETS}
            disabled={isBusy}
          />
          <p className="mt-1 text-xs text-gray-500">
            初回の既定値は {DEFAULT_MAX_TOKENS_FALLBACK.toLocaleString()}（エージェント向け）。131,072 超は数値入力のみ。
          </p>
        </div>
        )}

        {!isPoolingTask && (
        <>
        {/* 生成パラメータのデフォルト */}
        <div className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <FieldLabel
              label={`デフォルト temperature: ${form.default_temperature.toFixed(2)}`}
              hint="OpenAI API の sampling temperature。未指定時のみサーバーが既定として注入します。低いと決まった応答寄り、高いとランダム性が増えます。通常は 0.6〜1.0 付近。top_p と同時指定時は両方が効きます。"
            />
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              value={form.default_temperature}
              onChange={(e) => setForm({ ...form, default_temperature: parseFloat(e.target.value) })}
              disabled={isBusy}
              className="w-full accent-accent-primary"
            />
          </div>
          <div>
            <FieldLabel
              label={`デフォルト top_p: ${form.default_top_p.toFixed(2)}`}
              hint=" nucleus sampling。確率質量が上位 p までのトークンだけを候補にします。1.0 に近いほど広く、低いと尖った出力になります。temperature と併用されます。"
            />
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={form.default_top_p}
              onChange={(e) => setForm({ ...form, default_top_p: parseFloat(e.target.value) })}
              disabled={isBusy}
              className="w-full accent-accent-primary"
            />
          </div>
          <div>
            <FieldLabel
              label={`デフォルト frequency_penalty: ${form.default_frequency_penalty.toFixed(2)}`}
              hint="OpenAI 互換の frequency_penalty（概ね -2〜2）。これまでに出現したトークンを再度出しにくくします。正で繰り返し抑制、負で繰り返し増加の傾向。長文コピペの繰り返し対策に使うことが多いです。"
            />
            <input
              type="range"
              min="-2"
              max="2"
              step="0.05"
              value={form.default_frequency_penalty}
              onChange={(e) =>
                setForm({ ...form, default_frequency_penalty: parseFloat(e.target.value) })
              }
              disabled={isBusy}
              className="w-full accent-accent-primary"
            />
          </div>
          <div>
            <FieldLabel
              label={`デフォルト presence_penalty: ${form.default_presence_penalty.toFixed(2)}`}
              hint="OpenAI 互換の presence_penalty。既に出てきたトークン種類にペナルティを付け、新しい話題・語彙を促します。frequency は「回数」、presence は「出たかどうか」に効くイメージです。"
            />
            <input
              type="range"
              min="-2"
              max="2"
              step="0.05"
              value={form.default_presence_penalty}
              onChange={(e) =>
                setForm({ ...form, default_presence_penalty: parseFloat(e.target.value) })
              }
              disabled={isBusy}
              className="w-full accent-accent-primary"
            />
          </div>
        </div>
        </>
        )}

        {/* GPU メモリ利用率 */}
        <div className="mb-4">
          <FieldLabel
            label={`GPU メモリ利用率: ${form.gpu_memory_mode === "auto" ? "自動" : `${Math.round(form.gpu_memory_utilization * 100)}%`}`}
            hint="vLLM の `--gpu-memory-utilization` に対応します。GPU 全体 VRAM に対して「KV キャッシュ等に使ってよい上限比率」のイメージです。高いほど長いコンテキストや同時 seq を取りやすい一方、空きが少ないと起動失敗や他プロセスとの競合が起きやすくなります。自動は空き VRAM から算出します。"
          />
          <div className="flex gap-4 text-sm mb-2">
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="gpu_memory_mode"
                checked={form.gpu_memory_mode === "auto"}
                onChange={() => setForm({ ...form, gpu_memory_mode: "auto" })}
                disabled={isBusy}
              />
              自動（推奨）
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="gpu_memory_mode"
                checked={form.gpu_memory_mode === "manual"}
                onChange={() => setForm({ ...form, gpu_memory_mode: "manual" })}
                disabled={isBusy}
              />
              手動
            </label>
          </div>
          {form.gpu_memory_mode === "manual" ? (
            <>
              <NumberSliderField
                label="GPU メモリ利用率（0.1〜0.85）"
                hint="vLLM の `--gpu-memory-utilization` に対応。例: 0.75 と入力。"
                value={form.gpu_memory_utilization}
                onChange={(gpu_memory_utilization) =>
                  setForm({ ...form, gpu_memory_utilization })
                }
                min={0.1}
                max={0.85}
                step={0.05}
                integer={false}
                disabled={isBusy}
              />
              <p className="mt-1 text-xs text-gray-500">
                手動目安: 0.60〜0.75。起動失敗/OOM が出る場合はさらに下げてください。
              </p>
            </>
          ) : (
            <p className="mt-1 text-xs text-gray-500">
              起動時に空きVRAMから自動計算します。必要に応じて手動に切り替えて調整できます。
            </p>
          )}
        </div>

        {/* テンソル並列数 */}
        <NumberSliderField
          className="mb-4"
          label="テンソル並列数（tensor_parallel_size）"
          hint="vLLM の tensor-parallel-size。1 で単一 GPU、複数 GPU 時は 2,4,8 など（1〜8）。"
          value={form.tensor_parallel_size}
          onChange={(tensor_parallel_size) => setForm({ ...form, tensor_parallel_size })}
          min={1}
          max={8}
          step={1}
          disabled={isBusy}
        />

        {/* 使用GPU */}
        <div className="mb-4">
          <FieldLabel
            label={`使用GPU: ${form.gpu_devices}`}
            hint="起動時に環境変数 `CUDA_VISIBLE_DEVICES` として渡します。`all` はその制限なし（全GPUが見える状態）。`0` や `0,1` のようにすると、その番号の GPU だけがプロセスから見えます。別プロセスと GPU を分けたいときに指定します。"
          />
          <div className="flex gap-3 mb-2 text-sm">
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="gpu_device_mode"
                checked={form.gpu_devices === "all"}
                onChange={() => setForm({ ...form, gpu_devices: "all" })}
                disabled={isBusy}
              />
              全GPU
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="gpu_device_mode"
                checked={form.gpu_devices !== "all"}
                onChange={() =>
                  setForm({
                    ...form,
                    gpu_devices: gpuOptions.length ? String(gpuOptions[0].index) : "0",
                  })
                }
                disabled={isBusy}
              />
              選択したGPUのみ
            </label>
          </div>
          {form.gpu_devices !== "all" && (
            <div className="space-y-2">
              <div className="flex flex-wrap gap-2">
                {gpuOptions.map((gpu) => {
                  const selected = form.gpu_devices
                    .split(",")
                    .map((item) => item.trim())
                    .filter(Boolean);
                  const id = String(gpu.index);
                  const checked = selected.includes(id);
                  return (
                    <label
                      key={gpu.index}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-bg-tertiary border border-white/10 text-sm"
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={isBusy}
                        onChange={(e) => {
                          const next = e.target.checked
                            ? [...selected, id]
                            : selected.filter((value) => value !== id);
                          setForm({
                            ...form,
                            gpu_devices: next.length ? next.join(",") : id,
                          });
                        }}
                      />
                      GPU {gpu.index}
                    </label>
                  );
                })}
              </div>
              <input
                value={form.gpu_devices}
                onChange={(e) => setForm({ ...form, gpu_devices: e.target.value })}
                disabled={isBusy}
                className="w-full md:w-72 bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-accent-primary disabled:opacity-50"
                placeholder="例: 0,1"
              />
              {gpuOptions.length === 0 && (
                <p className="text-xs text-gray-500">
                  GPU一覧を取得できないため、番号は手入力してください（例: 0,1）。
                </p>
              )}
            </div>
          )}
        </div>

        {!isPoolingTask && (
        <>
        {/* Vision / multimodal */}
        <div className="mb-6 rounded-lg border border-white/10 bg-bg-tertiary/40 p-4 space-y-3">
          <FieldLabel
            label="Vision / マルチモーダル"
            hint="Qwen3.6 など image-text-to-text モデル向け。`--limit-mm-per-prompt` で 1 リクエストあたりの画像数上限を指定します。0 にすると Vision エンコーダが無効になります。LiteLLM 経由で画像付きリクエストを送るときは、下の「画像付きは stream 強制しない」もオン推奨です。"
          />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <FieldLabel
                label="1 プロンプトあたりの画像数上限"
                hint="vLLM の --limit-mm-per-prompt（例: image=1）。Qwen3.6 系は 1 から。0 で Vision 無効。"
              />
              <input
                type="number"
                min={0}
                max={8}
                value={form.limit_mm_image}
                onChange={(e) =>
                  setForm({
                    ...form,
                    limit_mm_image: Math.max(0, Math.min(8, Number(e.target.value) || 0)),
                  })
                }
                disabled={isBusy}
                className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              />
              {modelLikelySupportsVision(form.model_id) ? (
                <p className="text-xs text-gray-500 mt-1">このモデル ID は Vision 対応の可能性が高いです。</p>
              ) : null}
            </div>
            <div>
              <FieldLabel label="mm-encoder-tp-mode（任意）" hint="TP&gt;1 や高負荷時は data を検討。" />
              <select
                value={form.mm_encoder_tp_mode}
                onChange={(e) => setForm({ ...form, mm_encoder_tp_mode: e.target.value })}
                disabled={isBusy}
                className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              >
                <option value="">（デフォルト）</option>
                <option value="weights">weights</option>
                <option value="data">data</option>
              </select>
            </div>
            <div>
              <FieldLabel
                label="mm-processor-cache-type（任意）"
                hint="前処理キャッシュ。高負荷時は shm を検討。"
              />
              <select
                value={form.mm_processor_cache_type}
                onChange={(e) =>
                  setForm({ ...form, mm_processor_cache_type: e.target.value })
                }
                disabled={isBusy}
                className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              >
                <option value="">（デフォルト）</option>
                <option value="lru">lru</option>
                <option value="shm">shm</option>
              </select>
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={form.force_stream}
              onChange={(e) => setForm({ ...form, force_stream: e.target.checked })}
              disabled={isBusy}
            />
            LiteLLM 経由でテキストのみのとき <code className="text-xs bg-bg-primary px-1 rounded">stream: true</code>{" "}
            を強制（504 対策）。画像付きは自動でスキップされます。
          </label>
        </div>

        {/* Tool calling (tool_choice: auto) */}
        <div className="mb-6 rounded-lg border border-white/10 bg-bg-tertiary/40 p-4 space-y-3">
          <FieldLabel
            label="ツール呼び出し（auto tool choice）"
            hint="Hermes などが tool_choice: auto を送るとき vLLM が要求する `--enable-auto-tool-choice` と `--tool-call-parser` を付けます。パーサ名はモデルと vLLM のバージョンで異なります（利用可能な名前は `vllm serve --help` の説明を参照）。有効にした場合はパーサ名の入力が必須です。"
          />
          <label className="flex items-center gap-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={form.enable_auto_tool_choice}
              onChange={(e) => setForm({ ...form, enable_auto_tool_choice: e.target.checked })}
              disabled={isBusy}
            />
            <code className="text-xs bg-bg-primary px-1 rounded">--enable-auto-tool-choice</code>
            を付ける
          </label>
          <div>
            <FieldLabel
              label="tool-call-parser"
              hint="上記をオンにしたとき必須。空のままではサーバー起動を拒否します。"
            />
            <input
              type="text"
              value={form.tool_call_parser}
              onChange={(e) => setForm({ ...form, tool_call_parser: e.target.value })}
              disabled={isBusy}
              className="w-full max-w-xl bg-bg-tertiary border border-white/10 rounded-lg px-3 py-2 text-white placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-accent-primary disabled:opacity-50"
              placeholder="例: hermes / llama3_json（モデル・vLLM の --help で確認）"
            />
          </div>
        </div>

        {/* Speculative Decoding */}
        <div className="mb-6 rounded-lg border border-white/10 bg-bg-tertiary/40 p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <FieldLabel
              label="Speculative Decoding（speculative_config）"
              hint="vLLM の `--speculative-config` にそのまま渡す設定です。モデルに合う method を選ぶと token 生成の待ち時間を短縮できます。互換のない method を選ぶと起動失敗することがあります。まずは `ngram` / `suffix` から試し、MTP 対応モデルなら `mtp` / `qwen3_next_mtp` / `qwen3_5_mtp` を使うのが安全です。"
            />
            <label className="flex items-center gap-2 text-sm text-gray-300">
              <input
                type="checkbox"
                checked={specEnabled}
                onChange={(e) => setSpecEnabled(e.target.checked)}
                disabled={isBusy}
              />
              有効化
            </label>
          </div>
          <p className="text-xs text-gray-500">
            注: method とモデルの相性が最重要です。Qwen3.5 系で MTP を試す場合は `qwen3_next_mtp` か
            `qwen3_5_mtp` を選び、`num_speculative_tokens` は 1〜2 から始めてください。
          </p>

          {specEnabled && (
            <div className="space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                  <FieldLabel
                    label="method"
                    hint="推測デコード手法。`ngram`/`suffix` は比較的導入しやすく、`draft_model`/`eagle3` は追加モデルや互換条件が必要です。MTP 系 (`mtp`,`qwen3_next_mtp`,`qwen3_5_mtp`) はモデルがネイティブ対応している場合に使います。"
                  />
                  <select
                    value={specForm.method}
                    onChange={(e) =>
                      setSpecForm({ ...specForm, method: e.target.value as SpecMethod })
                    }
                    disabled={isBusy}
                    className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                  >
                    <option value="ngram">ngram</option>
                    <option value="suffix">suffix</option>
                    <option value="mtp">mtp</option>
                    <option value="qwen3_next_mtp">qwen3_next_mtp</option>
                    <option value="qwen3_5_mtp">qwen3_5_mtp</option>
                    <option value="eagle3">eagle3</option>
                    <option value="draft_model">draft_model</option>
                  </select>
                </div>
                <div>
                  <FieldLabel
                    label="num_speculative_tokens"
                    hint="1ステップで先読み提案する token 数です。大きすぎると却って効率が落ちることがあります。まず 1〜2（高くても 4 前後）から測定するのが一般的です。"
                  />
                  <input
                    type="number"
                    min={1}
                    value={specForm.num_speculative_tokens}
                    onChange={(e) =>
                      setSpecForm({
                        ...specForm,
                        num_speculative_tokens: Math.max(1, Number(e.target.value) || 1),
                      })
                    }
                    disabled={isBusy}
                    className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <FieldLabel
                    label="rejection_sample_method"
                    hint="提案 token の採択方式です。通常は `standard` を推奨。`synthetic` を使う場合は synthetic_acceptance_rate も指定します。"
                  />
                  <select
                    value={specForm.rejection_sample_method}
                    onChange={(e) =>
                      setSpecForm({
                        ...specForm,
                        rejection_sample_method: e.target.value as RejectionSampleMethod,
                      })
                    }
                    disabled={isBusy}
                    className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                  >
                    <option value="standard">standard</option>
                    <option value="synthetic">synthetic</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <FieldLabel
                    label="model（任意）"
                    hint="draft_model / eagle3 などで補助モデルを使う場合のモデル ID。MTP や ngram/suffix では通常不要です。"
                  />
                  <input
                    value={specForm.model}
                    onChange={(e) => setSpecForm({ ...specForm, model: e.target.value })}
                    disabled={isBusy}
                    placeholder="例: google/gemma-4-E4B-it-assistant"
                    className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <FieldLabel
                    label="max_model_len（任意）"
                    hint="補助モデル側の最大コンテキスト長。未指定なら vLLM 既定に任せます。draft モデルが短い場合に明示します。"
                  />
                  <input
                    type="number"
                    min={1}
                    value={specForm.max_model_len}
                    onChange={(e) =>
                      setSpecForm({ ...specForm, max_model_len: Math.max(1, Number(e.target.value) || 1) })
                    }
                    disabled={isBusy}
                    className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <FieldLabel
                    label="draft_tensor_parallel_size（任意）"
                    hint="補助モデル用 TP。target と同じにする必要はありません。補助モデルを軽く動かしたい場合に調整します。"
                  />
                  <input
                    type="number"
                    min={1}
                    value={specForm.draft_tensor_parallel_size}
                    onChange={(e) =>
                      setSpecForm({
                        ...specForm,
                        draft_tensor_parallel_size: Math.max(1, Number(e.target.value) || 1),
                      })
                    }
                    disabled={isBusy}
                    className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                  />
                </div>
                <div className="flex items-end">
                  <label className="flex items-center gap-2 text-sm text-gray-300">
                    <input
                      type="checkbox"
                      checked={specForm.parallel_drafting}
                      onChange={(e) =>
                        setSpecForm({ ...specForm, parallel_drafting: e.target.checked })
                      }
                      disabled={isBusy}
                    />
                    parallel_drafting
                    <InfoTooltip text="EAGLE / draft_model 系で提案 token を並列生成するオプションです。モデルとバージョン次第で非対応の場合があります。" />
                  </label>
                </div>
              </div>

              {specForm.rejection_sample_method === "synthetic" && (
                <div>
                  <FieldLabel
                    label={`synthetic_acceptance_rate: ${specForm.synthetic_acceptance_rate.toFixed(2)}`}
                    hint="rejection_sample_method を synthetic にしたときの採択率目標（0〜1）です。通常は 0.6〜0.8 付近から試します。"
                  />
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={specForm.synthetic_acceptance_rate}
                    onChange={(e) =>
                      setSpecForm({
                        ...specForm,
                        synthetic_acceptance_rate: Number(e.target.value),
                      })
                    }
                    disabled={isBusy}
                    className="w-full accent-accent-primary"
                  />
                </div>
              )}

              {specForm.method === "ngram" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <FieldLabel
                      label="prompt_lookup_min"
                      hint="ngram で見る最小窓長です。通常 2〜4 程度。"
                    />
                    <input
                      type="number"
                      min={1}
                      value={specForm.prompt_lookup_min}
                      onChange={(e) =>
                        setSpecForm({
                          ...specForm,
                          prompt_lookup_min: Math.max(1, Number(e.target.value) || 1),
                        })
                      }
                      disabled={isBusy}
                      className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <FieldLabel
                      label="prompt_lookup_max"
                      hint="ngram で見る最大窓長です。min 以上にしてください。通常 4〜8 程度。"
                    />
                    <input
                      type="number"
                      min={1}
                      value={specForm.prompt_lookup_max}
                      onChange={(e) =>
                        setSpecForm({
                          ...specForm,
                          prompt_lookup_max: Math.max(1, Number(e.target.value) || 1),
                        })
                      }
                      disabled={isBusy}
                      className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                </div>
              )}

              {specForm.method === "suffix" && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <FieldLabel
                      label="suffix_decoding_max_tree_depth"
                      hint="suffix 木探索の最大深さです。深いほど積極的に推測しますが計算量も増えます。"
                    />
                    <input
                      type="number"
                      min={1}
                      value={specForm.suffix_decoding_max_tree_depth}
                      onChange={(e) =>
                        setSpecForm({
                          ...specForm,
                          suffix_decoding_max_tree_depth: Math.max(1, Number(e.target.value) || 1),
                        })
                      }
                      disabled={isBusy}
                      className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <FieldLabel
                      label="suffix_decoding_max_cached_requests"
                      hint="suffix 共有キャッシュの件数上限です。0 で無効。"
                    />
                    <input
                      type="number"
                      min={0}
                      value={specForm.suffix_decoding_max_cached_requests}
                      onChange={(e) =>
                        setSpecForm({
                          ...specForm,
                          suffix_decoding_max_cached_requests: Math.max(0, Number(e.target.value) || 0),
                        })
                      }
                      disabled={isBusy}
                      className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <FieldLabel
                      label="suffix_decoding_max_spec_factor"
                      hint="prefix 一致長に対する推測長倍率の上限。1.0 付近から開始推奨。"
                    />
                    <input
                      type="number"
                      min={0.1}
                      step={0.1}
                      value={specForm.suffix_decoding_max_spec_factor}
                      onChange={(e) =>
                        setSpecForm({
                          ...specForm,
                          suffix_decoding_max_spec_factor: Math.max(0.1, Number(e.target.value) || 0.1),
                        })
                      }
                      disabled={isBusy}
                      className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <FieldLabel
                      label="suffix_decoding_min_token_prob"
                      hint="この確率未満の token は推測しない閾値（0〜1）です。上げると保守的になります。"
                    />
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={specForm.suffix_decoding_min_token_prob}
                      onChange={(e) =>
                        setSpecForm({
                          ...specForm,
                          suffix_decoding_min_token_prob: Math.max(
                            0,
                            Math.min(1, Number(e.target.value) || 0)
                          ),
                        })
                      }
                      disabled={isBusy}
                      className="w-full bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        </>
        )}

        {/* ダウンロード制御 */}
        <label className="mb-6 flex items-center gap-2 text-sm text-gray-400">
          <input
            type="checkbox"
            checked={form.download_model}
            onChange={(e) => setForm({ ...form, download_model: e.target.checked })}
            disabled={isBusy}
          />
          <span className="flex items-center gap-1">
            起動前にモデルキャッシュを確認/ダウンロードする
            <InfoTooltip text="オン時のみ、起動フローで Hugging Face の `snapshot_download` によりキャッシュを確認・未取得ならダウンロードします。オフにするとその確認をスキップします（既にキャッシュがある前提）。ゲート付きモデルは HF の利用許諾と `.env` の HF_TOKEN が必要です。初回のみ時間がかかります。" />
          </span>
        </label>

        {/* ボタン */}
        <div className="flex gap-3">
          <button
            onClick={handleStart}
            disabled={isBusy || !hasDownloadedModels}
            className="flex items-center gap-2 px-6 py-2.5 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            {action === "starting" ? "起動中..." : "起動"}
          </button>

          <button
            onClick={handleStop}
            disabled={isBusy || !status?.running}
            className="flex items-center gap-2 px-6 py-2.5 bg-accent-danger/20 hover:bg-accent-danger/30 text-accent-danger rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <StopCircle className="w-4 h-4" />
            {action === "stopping" ? "停止中..." : "default 停止"}
          </button>

          <button
            onClick={handleRestart}
            disabled={isBusy || !status?.running}
            className="flex items-center gap-2 px-6 py-2.5 bg-bg-tertiary hover:bg-bg-tertiary/80 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed border border-white/10"
          >
            <RotateCcw className="w-4 h-4" />
            {action === "restarting" ? "再起動中..." : "再起動"}
          </button>
        </div>
      </div>

      <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="text-lg font-semibold">起動中サーバー一覧</h3>
          <button
            onClick={handleCopyAllServerCommands}
            disabled={!runningServers.length}
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-bg-tertiary px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-bg-primary disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Copy className="h-3.5 w-3.5" />
            起動コマンドを全コピー
          </button>
        </div>
        <p className="text-xs text-gray-500 mb-3">
          このホストで動作中の `vllm serve` プロセスを表示します。複数起動時は一覧から個別に停止できます（LLM / embedding / rerank 混在可）。
        </p>
        {copyMessage && <p className="mb-3 text-xs text-gray-400">{copyMessage}</p>}
        <div className="overflow-x-auto rounded-lg border border-white/10">
          <table className="w-full min-w-[1100px] text-sm">
            <thead className="bg-bg-primary text-gray-400">
              <tr>
                <th className="px-3 py-2 text-left font-medium">PID</th>
                <th className="px-3 py-2 text-left font-medium">Instance</th>
                <th className="px-3 py-2 text-left font-medium">種別</th>
                <th className="px-3 py-2 text-left font-medium">モデル</th>
                <th className="px-3 py-2 text-left font-medium">Port</th>
                <th className="px-3 py-2 text-left font-medium">起動者/由来</th>
                <th className="px-3 py-2 text-left font-medium">起動設定</th>
                <th className="px-3 py-2 text-left font-medium">サーバー使用VRAM</th>
                <th className="px-3 py-2 text-left font-medium">GPU内訳</th>
                <th className="px-3 py-2 text-left font-medium">稼働時間</th>
                <th className="px-3 py-2 text-left font-medium">管理対象</th>
                <th className="px-3 py-2 text-left font-medium">操作</th>
              </tr>
            </thead>
            <tbody>
              {runningServers.map((server) => (
                <tr key={server.pid} className="border-t border-white/5 bg-bg-tertiary/40">
                  <td className="px-3 py-2 font-mono text-xs">{server.pid}</td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    {server.instance_name || server.instance_id || "-"}
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    {server.task_type === "embedding"
                      ? "embedding"
                      : server.task_type === "rerank"
                        ? "rerank"
                        : "LLM"}
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    {server.model || "-"}
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    {server.port ?? "-"}
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    <div>user: {server.owner || "-"}</div>
                    <div>ppid: {server.launcher_pid ?? "-"}</div>
                    <div className="text-gray-500 break-all">
                      parent: {server.launcher_cmd ? server.launcher_cmd.slice(0, 90) : "-"}
                    </div>
                    <div className="text-gray-500">
                      container: {server.container_id || "-"}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    <div>ctx: {server.context_length ?? "-"}</div>
                    <div>seqs: {server.max_num_seqs ?? "-"}</div>
                    <div>gpu util: {server.gpu_memory_utilization != null ? `${Math.round(server.gpu_memory_utilization * 100)}%` : "-"}</div>
                    <div>tp: {server.tensor_parallel_size ?? "-"}</div>
                    <div>CUDA_VISIBLE_DEVICES: {server.gpu_devices || "all"}</div>
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    <div>{server.vram_used_mb.toFixed(0)} MB</div>
                    <div className="text-gray-500">
                      GPU: {server.using_gpu_indices.length ? server.using_gpu_indices.join(",") : "-"}
                    </div>
                    {server.vram_estimated && (
                      <div className="text-[10px] text-gray-500">* GPU全体使用量からの推定値</div>
                    )}
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    {gpuMetrics.length === 0 ? (
                      <div className="text-gray-500">取得不可</div>
                    ) : (
                      gpuMetrics.map((gpu) => {
                        const serverGpuMb = server.vram_by_gpu_mb[String(gpu.index)];
                        return (
                          <div key={gpu.index} className="mb-1 last:mb-0">
                            GPU {gpu.index}: {gpu.utilization_percent.toFixed(0)}% /{" "}
                            {gpu.memory_used_mb.toFixed(0)}MB
                            {serverGpuMb != null ? (
                              <span className="text-accent-primary"> （このサーバー: {serverGpuMb.toFixed(0)}MB）</span>
                            ) : (
                              <span className="text-gray-500"> （このサーバー: -）</span>
                            )}
                          </div>
                        );
                      })
                    )}
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    {Math.floor(server.uptime_seconds / 60)}分
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-300">
                    {server.managed_by_app ? "このアプリ管理" : "外部起動"}
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex flex-col gap-1">
                      {server.instance_id && server.managed_by_app && (
                        <button
                          onClick={() => handleSmokeTest(server.instance_id!)}
                          disabled={smokeTestingId === server.instance_id}
                          className="px-3 py-1.5 rounded-lg bg-accent-primary/20 hover:bg-accent-primary/30 text-accent-primary disabled:opacity-50 text-xs"
                        >
                          {smokeTestingId === server.instance_id ? "疎通確認中..." : "疎通テスト"}
                        </button>
                      )}
                      {server.instance_id && smokeTestResults[server.instance_id] && (
                        <div
                          className={`text-[10px] rounded px-2 py-1 ${
                            smokeTestResults[server.instance_id].success
                              ? "bg-accent-success/10 text-accent-success"
                              : "bg-accent-danger/10 text-accent-danger"
                          }`}
                        >
                          {smokeTestResults[server.instance_id].success ? (
                            <>
                              OK {smokeTestResults[server.instance_id].latency_ms}ms
                              {smokeTestResults[server.instance_id].tokens_per_sec != null &&
                                ` / ${smokeTestResults[server.instance_id].tokens_per_sec} tok/s`}
                            </>
                          ) : (
                            smokeTestResults[server.instance_id].error || "失敗"
                          )}
                        </div>
                      )}
                      {server.instance_id && server.managed_by_app && (
                        <button
                          onClick={() => handleStopByInstance(server.instance_id!)}
                          disabled={stoppingInstanceId === server.instance_id}
                          className="px-3 py-1.5 rounded-lg bg-accent-danger/20 hover:bg-accent-danger/30 text-accent-danger disabled:opacity-50 text-xs"
                        >
                          {stoppingInstanceId === server.instance_id ? "停止中..." : "インスタンス停止"}
                        </button>
                      )}
                      <button
                        onClick={() => handleStopByPid(server.pid)}
                        disabled={stoppingPid === server.pid}
                        className="px-3 py-1.5 rounded-lg bg-accent-danger/20 hover:bg-accent-danger/30 text-accent-danger disabled:opacity-50 text-xs"
                      >
                        {stoppingPid === server.pid ? "停止中..." : "PID 停止"}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {runningServers.length === 0 && (
                <tr className="border-t border-white/5 bg-bg-tertiary/20">
                  <td colSpan={12} className="px-3 py-6 text-center text-gray-500">
                    起動中の vLLM サーバーはありません
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 実行結果 */}
      {result && showSteps && (
        <div className="bg-bg-secondary rounded-xl border border-white/5 p-6">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                result.success ? "bg-accent-success" : "bg-accent-danger"
              }`}
            />
            {result.success ? "成功" : "エラー"}
          </h3>
          <div className="bg-bg-primary rounded-lg p-3 font-mono text-xs text-gray-300 space-y-1">
            {result.steps?.map((step, i) => (
              <div key={i} className="animate-slide-in" style={{ animationDelay: `${i * 100}ms` }}>
                <span className="text-accent-primary">{">"}</span> {step}
              </div>
            ))}
            {result.message && !result.steps?.length && (
              <div>{result.message}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function NumberSliderField({
  className = "",
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step = 1,
  integer = true,
  disabled,
  showSlider = true,
  sliderMin,
  sliderMax,
  sliderStep,
  sliderScale = "linear",
  sliderHint,
  presets,
  quickValues,
}: {
  className?: string;
  label: string;
  hint: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  integer?: boolean;
  disabled?: boolean;
  showSlider?: boolean;
  sliderMin?: number;
  sliderMax?: number;
  sliderStep?: number;
  sliderScale?: "linear" | "log";
  sliderHint?: string;
  presets?: NumberPreset[];
  quickValues?: number[];
}) {
  const sMin = sliderMin ?? min;
  const sMax = sliderMax ?? max;
  const sStep = sliderStep ?? step;
  const isLog = sliderScale === "log" && sMin > 0 && sMax > sMin;

  const linearSliderValue = clampNumber(value, sMin, sMax);
  const logSliderPos = isLog ? valueToLogSliderPos(value, sMin, sMax) : 0;

  const apply = (next: number) => onChange(clampNumber(next, min, max));

  return (
    <div className={className}>
      <FieldLabel label={label} hint={hint} />
      <div className="max-w-2xl space-y-2">
        {presets && presets.length > 0 && (
          <div>
            <p className="mb-1.5 text-xs text-gray-500">プリセット</p>
            <div className="flex flex-wrap gap-2">
              {presets.map((preset) => {
                const selected = value === preset.value;
                return (
                  <button
                    key={`${preset.label}-${preset.value}`}
                    type="button"
                    disabled={disabled}
                    title={preset.hint}
                    onClick={() => apply(preset.value)}
                    className={`flex min-w-[5.5rem] flex-col items-start rounded-lg border px-3 py-2 text-left transition-colors ${
                      selected
                        ? "border-accent-primary bg-accent-primary/15 text-white"
                        : "border-white/10 bg-bg-tertiary text-gray-300 hover:border-white/20 hover:text-white"
                    } disabled:opacity-50`}
                  >
                    <span className="text-xs font-semibold">{preset.label}</span>
                    <span className={`text-sm tabular-nums ${selected ? "text-accent-primary" : "text-gray-400"}`}>
                      {preset.value.toLocaleString()}
                    </span>
                    {preset.hint && (
                      <span className="mt-0.5 text-[10px] leading-tight text-gray-500">{preset.hint}</span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => {
            const parsed = parseNumberInput(e.target.value, integer);
            if (parsed !== null) apply(parsed);
          }}
          onBlur={(e) => {
            const parsed = parseNumberInput(e.target.value, integer);
            if (parsed === null) apply(value);
          }}
          className={INPUT_CLASS}
        />
        {showSlider && (
          <>
            <input
              type="range"
              min={isLog ? 0 : sMin}
              max={isLog ? LOG_SLIDER_STEPS : sMax}
              step={isLog ? 1 : sStep}
              value={isLog ? logSliderPos : linearSliderValue}
              disabled={disabled}
              onChange={(e) => {
                const parsed = integer
                  ? parseInt(e.target.value, 10)
                  : parseFloat(e.target.value);
                if (!Number.isFinite(parsed)) return;
                if (isLog) {
                  apply(logSliderPosToValue(parsed, sMin, sMax, sStep));
                } else {
                  apply(parsed);
                }
              }}
              className={SLIDER_CLASS}
            />
            <div className="flex justify-between text-xs text-gray-500 tabular-nums">
              <span>{sMin.toLocaleString()}</span>
              <span>{sliderHint ?? (isLog ? "対数スケール" : "")}</span>
              <span>{sMax.toLocaleString()}</span>
            </div>
          </>
        )}
        {quickValues && quickValues.length > 0 && !presets?.length && (
          <div className="flex flex-wrap gap-2">
            {quickValues.map((qv) => (
              <button
                key={qv}
                type="button"
                disabled={disabled}
                onClick={() => apply(qv)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium border transition-colors ${
                  value === qv
                    ? "bg-accent-primary text-white border-accent-primary"
                    : "bg-bg-tertiary text-gray-400 border-white/10 hover:text-white"
                } disabled:opacity-50`}
              >
                {qv.toLocaleString()}
              </button>
            ))}
          </div>
        )}
        {showSlider && value > sMax && (
          <p className="text-xs text-gray-500">
            スライダーは {sMax.toLocaleString()} まで。現在 {value.toLocaleString()}（数値入力で指定中）
          </p>
        )}
      </div>
    </div>
  );
}



function FieldLabel({ label, hint }: { label: string; hint: string }) {
  return (
    <label className="mb-1 flex items-center gap-1 text-sm text-gray-400">
      {label}
      <InfoTooltip text={hint} />
    </label>
  );
}

function InfoTooltip({ text }: { text: ReactNode }) {
  return (
    <span className="group relative inline-flex">
      <Info className="h-4 w-4 cursor-help text-gray-500 transition-colors group-hover:text-accent-primary" />
      <span className="pointer-events-none absolute left-1/2 top-6 z-30 hidden w-80 -translate-x-1/2 rounded-lg border border-white/10 bg-bg-primary p-3 text-xs font-normal leading-relaxed text-gray-300 shadow-xl group-hover:block">
        {text}
      </span>
    </span>
  );
}
