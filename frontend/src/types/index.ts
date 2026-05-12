// タイプ定義

export interface ServerStatus {
  running: boolean;
  healthy: boolean;
  pid: number | null;
  vllm_port: number;
  model: string | null;
  uptime_seconds: number;
}

export interface RunningServer {
  pid: number;
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
  required_gpu_memory_gb?: number | null;
  allowed_roles?: string[];
  source?: string;
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
}

export interface ApiResponse {
  success: boolean;
  message: string;
  steps?: string[];
}

export interface MetricsData {
  timestamp: number;
  gpu_memory_usage_gb: number;
  gpu_cpu_total_gb: number;
  num_requests_running: number;
  num_requests_waiting: number;
  num_requests_swapped: number;
  iteration_tokens: number;
  time_per_output_token_ms: number;
  request_throughput_rps: number;
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
  error: string | null;
  started_at: number;
  updated_at: number;
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
  gpus: SystemGpuMetrics[];
  gpu_processes: GpuProcessMetrics[];
}
