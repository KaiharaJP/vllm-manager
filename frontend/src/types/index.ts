// タイプ定義

export interface ServerStatus {
  running: boolean;
  healthy: boolean;
  pid: number | null;
  vllm_port: number;
  model: string | null;
  uptime_seconds: number;
  instance_id?: string | null;
  task_type?: "chat" | "embedding" | "rerank";
}

export interface RunningServer {
  pid: number;
  instance_id?: string | null;
  instance_name?: string | null;
  task_type?: "chat" | "embedding" | "rerank";
  model: string | null;
  port: number | null;
  context_length: number | null;
  max_num_seqs: number | null;
  gpu_memory_utilization: number | null;
  tensor_parallel_size: number | null;
  gpu_devices: string;
  vram_used_mb: number;
  vram_by_gpu_mb: Record<string, number>;
  vram_estimated: boolean;
  using_gpu_indices: number[];
  uptime_seconds: number;
  managed_by_app: boolean;
  owner?: string | null;
  launcher_pid?: number | null;
  launcher_cmd?: string | null;
  container_id?: string | null;
  cgroup_path?: string | null;
  command: string;
}

export interface Model {
  id: string;
  name: string;
  size: string;
  revision?: string | null;
  gated?: boolean;
  trust_remote_code?: boolean;
  recommended_context_length?: number;
  output_dimension?: number | null;
  license_note?: string | null;
  required_gpu_memory_gb?: number | null;
  allowed_roles?: string[];
  source?: string;
  task_type?: "chat" | "embedding" | "rerank";
  downloaded?: boolean;
  cache_path?: string | null;
  cache_size_bytes?: number;
}

export interface ContextPreset {
  value: number;
  label: string;
}

export interface ServerConfig {
  model_id: string;
  context_length: number;
  max_num_seqs: number;
  default_max_tokens: number;
  default_temperature: number;
  default_top_p: number;
  default_frequency_penalty: number;
  default_presence_penalty: number;
  gpu_memory_mode: "auto" | "manual";
  gpu_memory_utilization: number;
  tensor_parallel_size: number;
  gpu_devices: string;
  speculative_config?: Record<string, unknown> | null;
  vllm_port: number;
  enable_auto_tool_choice?: boolean;
  tool_call_parser?: string;
  force_stream?: boolean;
  limit_mm_per_prompt?: Record<string, number> | null;
  mm_encoder_tp_mode?: string;
  mm_processor_cache_type?: string;
  task_type?: "chat" | "embedding" | "rerank";
  trust_remote_code?: boolean;
  instance_id?: string | null;
  instance_name?: string | null;
}

export interface ServerInstance {
  instance_id: string;
  instance_name?: string | null;
  task_type?: "chat" | "embedding" | "rerank";
  running: boolean;
  healthy: boolean;
  pid: number | null;
  vllm_port: number;
  model: string | null;
  uptime_seconds: number;
  model_id?: string;
  started_at?: number;
}

export interface ServerStartRequest {
  model_id: string;
  context_length: number;
  max_num_seqs: number;
  default_max_tokens: number;
  default_temperature: number;
  default_top_p: number;
  default_frequency_penalty: number;
  default_presence_penalty: number;
  gpu_memory_mode: "auto" | "manual";
  gpu_memory_utilization: number;
  tensor_parallel_size: number;
  gpu_devices: string;
  speculative_config: Record<string, unknown>;
  download_model: boolean;
  enable_auto_tool_choice: boolean;
  tool_call_parser: string;
  force_stream: boolean;
  limit_mm_per_prompt?: Record<string, number> | null;
  mm_encoder_tp_mode?: string;
  mm_processor_cache_type?: string;
  task_type?: "chat" | "embedding" | "rerank";
  trust_remote_code?: boolean;
  instance_id?: string | null;
  instance_name?: string | null;
  create_new_instance?: boolean;
}

export interface ApiResponse {
  success: boolean;
  message: string;
  steps?: string[];
  instance_id?: string;
}

export interface ApiToken {
  id: string;
  name: string;
  username: string;
  prefix: string;
  created_at: number;
  last_used_at: number | null;
  expires_at: number | null;
  disabled: boolean;
}

export interface ApiTokenCreated extends ApiToken {
  token: string;
}

export interface SmokeTestResult {
  instance_id: string;
  success: boolean;
  task_type: "chat" | "embedding" | "rerank" | null;
  latency_ms: number | null;
  tokens_generated: number | null;
  tokens_per_sec: number | null;
  response_preview: string | null;
  error: string | null;
}

export interface MetricsData {
  timestamp: number;
  gpu_memory_usage_gb: number;
  gpu_cpu_total_gb: number;
  num_requests_running: number;
  num_requests_waiting: number;
  num_requests_waiting_capacity: number;
  num_requests_waiting_deferred: number;
  num_requests_swapped: number;
  iteration_tokens: number;
  time_per_output_token_ms: number;
  request_throughput_rps: number;
  prompt_throughput_tok_s: number;
  generation_throughput_tok_s: number;
  kv_cache_usage_perc: number;
  gpu_compute_time: number;
}

export interface AppUser {
  username: string;
  role: "admin" | "user";
  litellm_user_id?: string | null;
  litellm_team_id?: string | null;
  disabled?: boolean;
  created_at?: number;
  security_warnings?: string[];
  must_change_password?: boolean;
}

export interface DownloadJob {
  id: string;
  model_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  downloaded_bytes: number;
  total_bytes: number;
  current_file?: string | null;
  message?: string | null;
  error?: string | null;
  error_code?: string | null;
  error_hint?: string | null;
  created_at: number;
  updated_at: number;
  actor?: string | null;
}

/** LiteLLM → backend /v1 経由で追跡中の 1 リクエスト（WebSocket `litellm_proxy_request`） */
export interface LiteLLMProxyRequestRow {
  id: string;
  endpoint: string;
  model: string;
  stream: boolean;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  completion_chunks: number;
  status: "streaming" | "pending" | "completed" | "error";
  phase: "prefill" | "generate" | "done" | "error";
  first_token_at: number | null;
  prefill_tok_s: number | null;
  gen_tok_s: number | null;
  elapsed_s: number;
  error: string | null;
  started_at: number;
  updated_at: number;
  max_tokens?: number | null;
  message_count?: number;
  prompt_char_est?: number;
  request_summary?: string;
  messages_truncated?: boolean;
}

export interface ChatMessage {
  role?: string;
  content?: string | unknown;
}

export interface ChatUiMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatModelsResponse {
  models: string[];
  count: number;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatUiMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
}

export interface LiteLLMProxyRequestDetail extends LiteLLMProxyRequestRow {
  messages?: ChatMessage[];
  messages_truncated?: boolean;
  completed_at?: number;
}

export interface RequestHistoryListResponse {
  requests: LiteLLMProxyRequestDetail[];
  total: number;
  limit: number;
  offset: number;
}

export interface AppEvent {
  id?: string;
  type:
    | "metrics"
    | "history"
    | "event_history"
    | "model_download"
    | "model_registered"
    | "server_job"
    | "user_updated"
    | "litellm_key_updated"
    | "litellm_user_updated"
    | "litellm_team_updated"
    | "litellm_proxy_request"
    | "litellm_proxy_snapshot"
    | "metrics_scrape_error"
    | "error"
    | "pong";
  timestamp?: number;
  data?: MetricsData | MetricsData[] | DownloadJob | DownloadJob[] | AppEvent[] | unknown;
  message?: string;
  actor?: string | null;
}

export interface MetricsMessage {
  type: "metrics" | "history" | "event_history" | "litellm_proxy_request" | "litellm_proxy_snapshot" | "error" | "pong";
  data?: MetricsData | MetricsData[] | LiteLLMProxyRequestRow | { requests: LiteLLMProxyRequestRow[] };
  message?: string;
}

export interface LiteLLMStatus {
  healthy: boolean;
  url?: string;
  check?: string;
}

export interface ServiceHealthCheckResult {
  checked_at: number;
  method: string;
  vllm: {
    healthy: boolean;
    running: boolean;
    port: number;
    model: string | null;
    message: string;
  };
  litellm: {
    liveliness: boolean;
    readiness: boolean;
    url: string;
    liveliness_detail?: string | null;
    readiness_detail?: unknown;
  };
  backend: {
    healthy: boolean;
    message: string;
  };
}

export interface SystemGpuMetrics {
  index: number;
  name: string;
  utilization_percent: number;
  memory_used_mb: number;
  memory_total_mb: number;
  temperature_c: number;
}

export interface GpuProcessMetrics {
  pid: number;
  process_name: string;
  used_memory_mb: number;
  gpu_uuid: string;
  gpu_index: number | null;
  cmdline: string;
}

export interface SystemMetrics {
  cpu: {
    usage_percent: number;
    cores_logical: number;
    cores_physical: number | null;
  };
  memory: {
    usage_percent: number;
    used_gb: number;
    total_gb: number;
  };
  disk: {
    usage_percent: number;
    used_gb: number;
    total_gb: number;
  };
  disks: Array<{
    label: string;
    path: string;
    usage_percent: number;
    used_gb: number;
    total_gb: number;
  }>;
  gpus: SystemGpuMetrics[];
  gpu_processes: GpuProcessMetrics[];
}

// --- ストレージ使用状況 ---

export interface StorageMount {
  label: string;
  path: string;
  total_gb: number;
  used_gb: number;
  free_gb: number;
  used_percent: number | null;
  same_device_as_above: boolean;
}

export interface StorageBreakdownEntry {
  path: string;
  size_gb: number;
}

export interface StorageBreakdown {
  success: boolean;
  path: string;
  total_gb: number | null;
  entries: StorageBreakdownEntry[];
  entries_omitted: number;
  scanned_at: number;
  warnings: string[];
  cached?: boolean;
  age_sec?: number;
}

export interface StorageUsageItem {
  name: string;
  size_gb: number;
}

export interface StorageUsageSection {
  category: string;
  path: string;
  total_gb: number | null;
  items: StorageUsageItem[];
  note?: string;
}

export interface StorageUsageReport {
  success: boolean;
  sections: StorageUsageSection[];
  scanned_at: number;
  cached?: boolean;
  age_sec?: number;
}
