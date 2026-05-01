"""
vLLM Manager - メトリクススクラッパー

vLLM の Prometheus metrics endpoint をスクラップして
WebSocket 経由でリアルタイムデータを配信する。
"""

import asyncio
import time
import httpx
from typing import Dict, List, Optional
from collections import deque


class MetricsScraper:
    """vLLM Prometheus metrics のスクラッパー。"""

    def __init__(self, vllm_metrics_url: str = "http://localhost:8001/metrics", scrape_interval: float = 5.0):
        self.metrics_url = vllm_metrics_url
        self.scrape_interval = scrape_interval
        self.clients: List[asyncio.websocket.WebSocket] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._history = deque(maxlen=100)  # 直近100回のメトリクス履歴

    def register_client(self, ws) -> None:
        if ws not in self.clients:
            self.clients.append(ws)

    def unregister_client(self, ws) -> None:
        if ws in self.clients:
            self.clients.remove(ws)

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
                    await self._broadcast(metrics)
            except Exception as e:
                error_msg = {"type": "error", "message": f"Metrics scrape failed: {str(e)}"}
                await self._broadcast(error_msg)
            await asyncio.sleep(self.scrape_interval)

    async def _fetch_metrics(self) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self.metrics_url)
                if resp.status_code == 200:
                    return self._parse_prometheus_text(resp.text)
        except httpx.ConnectError:
            pass  # vLLM が起動していない場合はスキップ
        return None

    def _parse_prometheus_text(self, text: str) -> dict:
        """Prometheus text format を簡易パースする。"""
        metrics = {
            "timestamp": time.time(),
            "gpu_memory_usage_gb": 0,
            "gpu_cpu_total_gb": 0,
            "num_requests_running": 0,
            "num_requests_waiting": 0,
            "num_requests_swapped": 0,
            "iteration_tokens": 0,
            "iteration_tokens_hist": [],
            "time_per_output_token_ms": 0,
            "time_per_output_token_hist": [],
            "request_throughput_rps": 0,
            "kv_cache_usage_perc": 0,
            "prefix_cache_hit_rate": 0,
            "gpu_compute_time": 0,
        }

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

            # リクエスト数
            elif "num_requests_running" in metric_name:
                metrics["num_requests_running"] = int(value)
            elif "num_requests_waiting" in metric_name:
                metrics["num_requests_waiting"] = int(value)
            elif "num_requests_swapped" in metric_name:
                metrics["num_requests_swapped"] = int(value)

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

        return metrics

    async def _broadcast(self, data: dict) -> None:
        """全クライアントにデータをブロードキャストする。"""
        dead_clients = []
        for client in self.clients:
            try:
                await client.send_json(data)
            except Exception:
                dead_clients.append(client)

        for dead in dead_clients:
            self.unregister_client(dead)

    def get_history(self, count: int = 20) -> list:
        return list(self._history)[-count:]
