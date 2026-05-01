// タイプ定義

export interface ServerStatus {
  running: boolean;
  healthy: boolean;
  pid: number | null;
  vllm_port: number;
  model: string | null;
  uptime_seconds: number;
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
  gpu_memory_utilization: number;
  tensor_parallel_size: number;
  vllm_port: number;
}

export interface ServerStartRequest {
  model_id: string;
  context_length: number;
  max_num_seqs: number;
  gpu_memory_utilization: number;
  tensor_parallel_size: number;
  download_model: boolean;
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
  created_at: number;
  updated_at: number;
  actor?: string | null;
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
    | "error"
    | "pong";
  timestamp?: number;
  data?: MetricsData | MetricsData[] | DownloadJob | DownloadJob[] | AppEvent[] | unknown;
  message?: string;
  actor?: string | null;
}

export interface MetricsMessage {
  type: "metrics" | "history" | "event_history" | "error" | "pong";
  data?: MetricsData | MetricsData[];
  message?: string;
}

export interface LiteLLMStatus {
  healthy: boolean;
}
