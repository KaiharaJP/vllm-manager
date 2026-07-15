"""ホストのインフラリソース（GPU温度・VRAM・ディスク）のしきい値監視。

vLLM プロセス自体のヘルス（`app/health_watchdog.py`）とは別に、GPU が
危険な温度に達している、VRAM/ディスクが逼迫している、といった「サーバー
自体が壊れる/止まる前触れ」を検知して通知する。自動対処は行わず、
通知のみ（`instance_health_watchdog` と同じ思想）。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable, Optional

DEFAULT_GPU_TEMP_ALERT_C = float(os.environ.get("VLLM_MANAGER_GPU_TEMP_ALERT_C", "85"))
DEFAULT_GPU_MEMORY_ALERT_PERCENT = float(
    os.environ.get("VLLM_MANAGER_GPU_MEMORY_ALERT_PERCENT", "95")
)
DEFAULT_DISK_ALERT_PERCENT = float(os.environ.get("VLLM_MANAGER_DISK_ALERT_PERCENT", "90"))
DEFAULT_CHECK_INTERVAL_SEC = float(os.environ.get("VLLM_MANAGER_RESOURCE_CHECK_INTERVAL_SEC", "30"))

# しきい値ちょうど付近での通知/回復の連打（フラッピング）を防ぐためのマージン
RECOVERY_MARGIN = 5.0


class ResourceWatchdog:
    def __init__(
        self,
        *,
        check_interval: float = DEFAULT_CHECK_INTERVAL_SEC,
        gpu_temp_threshold_c: float = DEFAULT_GPU_TEMP_ALERT_C,
        gpu_memory_threshold_percent: float = DEFAULT_GPU_MEMORY_ALERT_PERCENT,
        disk_threshold_percent: float = DEFAULT_DISK_ALERT_PERCENT,
        event_publisher: Optional[Callable[..., Awaitable[dict]]] = None,
        metrics_provider: Optional[Callable[[], dict[str, Any]]] = None,
    ):
        self.check_interval = check_interval
        self.gpu_temp_threshold_c = gpu_temp_threshold_c
        self.gpu_memory_threshold_percent = gpu_memory_threshold_percent
        self.disk_threshold_percent = disk_threshold_percent
        self.event_publisher = event_publisher
        self.metrics_provider = metrics_provider
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._notified: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._check_once()
            except Exception:
                pass
            await asyncio.sleep(self.check_interval)

    def _get_metrics(self) -> dict[str, Any]:
        if self.metrics_provider:
            return self.metrics_provider()
        from app.system_metrics import get_system_metrics

        return get_system_metrics()

    async def _check_once(self) -> None:
        metrics = self._get_metrics()
        current_keys: set[str] = set()

        for gpu in metrics.get("gpus", []):
            index = gpu.get("index")
            temp = gpu.get("temperature_c")
            mem_used = gpu.get("memory_used_mb") or 0
            mem_total = gpu.get("memory_total_mb") or 0
            mem_percent = (mem_used / mem_total * 100) if mem_total else 0

            if temp is not None:
                key = f"gpu:{index}:temp"
                current_keys.add(key)
                await self._evaluate(
                    key=key,
                    value=float(temp),
                    threshold=self.gpu_temp_threshold_c,
                    label=f"GPU {index} 温度",
                    unit="C",
                )

            key = f"gpu:{index}:memory"
            current_keys.add(key)
            await self._evaluate(
                key=key,
                value=mem_percent,
                threshold=self.gpu_memory_threshold_percent,
                label=f"GPU {index} VRAM使用率",
                unit="%",
            )

        for disk in metrics.get("disks", []):
            label = disk.get("label", disk.get("path", "disk"))
            key = f"disk:{label}"
            current_keys.add(key)
            await self._evaluate(
                key=key,
                value=float(disk.get("usage_percent") or 0),
                threshold=self.disk_threshold_percent,
                label=f"ディスク使用率（{label}）",
                unit="%",
            )

        for stale_key in list(self._notified.keys()):
            if stale_key not in current_keys:
                self._notified.pop(stale_key, None)

    async def _evaluate(
        self, *, key: str, value: float, threshold: float, label: str, unit: str
    ) -> None:
        already_notified = key in self._notified
        if value >= threshold and not already_notified:
            self._notified[key] = {"value": value, "threshold": threshold}
            await self._publish(
                "resource_alert",
                {"resource": key, "label": label, "value": value, "threshold": threshold, "unit": unit},
                message=f"{label} がしきい値を超えています: {value:.1f}{unit}（しきい値 {threshold:.1f}{unit}）",
            )
        elif already_notified and value < threshold - RECOVERY_MARGIN:
            self._notified.pop(key, None)
            await self._publish(
                "resource_alert_recovered",
                {"resource": key, "label": label, "value": value, "threshold": threshold, "unit": unit},
                message=f"{label} がしきい値未満に戻りました: {value:.1f}{unit}",
            )

    async def _publish(self, event_type: str, data: Any, *, message: str) -> None:
        if self.event_publisher:
            await self.event_publisher(event_type, data, message=message, actor="system")
