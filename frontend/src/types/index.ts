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

export interface MetricsMessage {
  type: "metrics" | "history" | "error" | "pong";
  data?: MetricsData | MetricsData[];
  message?: string;
}

export interface LiteLLMStatus {
  healthy: boolean;
}
