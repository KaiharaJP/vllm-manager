"""
vLLM Manager - メトリクススクラッパー

vLLM の Prometheus metrics endpoint をスクラップして
WebSocket 経由でリアルタイムデータを配信する。
"""

import asyncio
import time
import httpx
from typing import Awaitable, Callable, Optional
from collections import deque


class MetricsScraper:
    """vLLM Prometheus metrics のスクラッパー。"""

    def __init__(
        self,
        vllm_metrics_url: str = "http://localhost:8001/metrics",
        scrape_interval: float = 5.0,
        event_publisher: Callable[..., Awaitable[dict]] | None = None,
    ):
        self.metrics_url = vllm_metrics_url
        self.scrape_interval = scrape_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._history = deque(maxlen=100)  # 直近100回のメトリクス履歴
        self.event_publisher = event_publisher
        self._prev_counters: dict[str, float] | None = None
        self._prev_timestamp: float | None = None

    async def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._scrape_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _scrape_loop(self) -> None:
        while self._running:
            try:
                metrics = await self._fetch_metrics()
                if metrics:
                    self._history.append(metrics)
                    await self._publish("metrics", metrics)
            except Exception as e:
                url = self._resolve_metrics_url()
                if url:
                    await self._publish(
                        "metrics_scrape_error",
                        message=f"Metrics scrape failed: {e!s}".strip() or type(e).__name__,
                    )
            await asyncio.sleep(self.scrape_interval)

    def _resolve_metrics_url(self) -> Optional[str]:
        """実行中 vLLM の実ポートを server_manager から解決する。"""
        try:
            from app.server_manager import get_status, load_config

            status = get_status()
            if not status.get("running"):
                return None
            port = int(status.get("vllm_port") or load_config().get("vllm_port", 8001))
            return f"http://localhost:{port}/metrics"
        except Exception:
            return self.metrics_url

    async def _fetch_metrics(self) -> Optional[dict]:
        url = self._resolve_metrics_url()
        if not url:
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return self._parse_prometheus_text(resp.text)
        except httpx.ConnectError:
            pass  # vLLM が起動していない／ポート未リッスン
        except Exception:
            raise
        return None

    def _parse_prometheus_text(self, text: str) -> dict:
        """Prometheus text format を簡易パースする。"""
        now = time.time()
        metrics = {
            "timestamp": now,
            "gpu_memory_usage_gb": 0,
            "gpu_cpu_total_gb": 0,
            "num_requests_running": 0,
            "num_requests_waiting": 0,
            "num_requests_waiting_capacity": 0,
            "num_requests_waiting_deferred": 0,
            "num_requests_swapped": 0,
            "iteration_tokens": 0,
            "iteration_tokens_hist": [],
            "time_per_output_token_ms": 0,
            "time_per_output_token_hist": [],
            "request_throughput_rps": 0,
            "prompt_throughput_tok_s": 0.0,
            "generation_throughput_tok_s": 0.0,
            "kv_cache_usage_perc": 0,
            "prefix_cache_hit_rate": 0,
            "gpu_compute_time": 0,
        }
        raw_counters: dict[str, float] = {}

        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue

            metric_name = parts[0]
            try:
                value = float(parts[1])
            except ValueError:
                continue

            # GPU メモリ使用量
            if "gpu_memory_usage" in metric_name:
                metrics["gpu_memory_usage_gb"] = value / (1024 ** 3)
            elif "gpu_memory_total" in metric_name:
                metrics["gpu_cpu_total_gb"] = value / (1024 ** 3)

            # リクエスト数（waiting_by_reason は waiting より先に判定）
            elif "num_requests_running" in metric_name and "waiting" not in metric_name:
                metrics["num_requests_running"] = int(value)
            elif "num_requests_waiting_by_reason" in metric_name:
                if 'reason="capacity"' in metric_name:
                    metrics["num_requests_waiting_capacity"] = int(value)
                elif 'reason="deferred"' in metric_name:
                    metrics["num_requests_waiting_deferred"] = int(value)
            elif metric_name.startswith("vllm:num_requests_waiting") or (
                "num_requests_waiting" in metric_name and "by_reason" not in metric_name
            ):
                metrics["num_requests_waiting"] = int(value)
            elif "num_requests_swapped" in metric_name:
                metrics["num_requests_swapped"] = int(value)

            # エンジン合計トークン（差分で tok/s を算出）
            elif metric_name.startswith("vllm:prompt_tokens_total") and "by_source" not in metric_name:
                raw_counters["prompt_tokens_total"] = value
            elif metric_name.startswith("vllm:generation_tokens_total"):
                raw_counters["generation_tokens_total"] = value

            # トークン数
            elif "iteration_tokens" in metric_name and "avg" in metric_name:
                metrics["iteration_tokens"] = int(value)
            elif "iteration_tokens" in metric_name:
                metrics["iteration_tokens_hist"].append(value)

            # トークン処理時間
            elif "time_per_output_token" in metric_name and "avg" in metric_name:
                metrics["time_per_output_token_ms"] = value * 1000
            elif "time_per_output_token" in metric_name:
                metrics["time_per_output_token_hist"].append(value * 1000)

            # スループット
            elif "request_throughput" in metric_name:
                metrics["request_throughput_rps"] = value

            # KV キャッシュ
            elif "kv_cache_usage_perc" in metric_name:
                metrics["kv_cache_usage_perc"] = value * 100

            # プレフィックスキャッシュ
            elif "prefix_cache_hit_rate" in metric_name:
                metrics["prefix_cache_hit_rate"] = value * 100

            # GPU 計算時間
            elif "gpu_compute_time" in metric_name:
                metrics["gpu_compute_time"] = value

        if raw_counters and self._prev_counters is not None and self._prev_timestamp is not None:
            dt = now - self._prev_timestamp
            if dt > 0:
                prev_pt = self._prev_counters.get("prompt_tokens_total")
                prev_gt = self._prev_counters.get("generation_tokens_total")
                if prev_pt is not None and "prompt_tokens_total" in raw_counters:
                    metrics["prompt_throughput_tok_s"] = max(
                        0.0, (raw_counters["prompt_tokens_total"] - prev_pt) / dt
                    )
                if prev_gt is not None and "generation_tokens_total" in raw_counters:
                    metrics["generation_throughput_tok_s"] = max(
                        0.0, (raw_counters["generation_tokens_total"] - prev_gt) / dt
                    )

        if raw_counters:
            self._prev_counters = raw_counters
            self._prev_timestamp = now

        return metrics

    async def _publish(self, event_type: str, data: dict | None = None, message: str | None = None) -> None:
        if self.event_publisher:
            await self.event_publisher(event_type, data, message=message)

    def get_history(self, count: int = 20) -> list:
        return list(self._history)[-count:]
