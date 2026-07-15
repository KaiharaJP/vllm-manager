"""稼働中 vLLM インスタンスの受動的ヘルス監視。

自動再起動は行わない（意図しないループ再起動によるリソース浪費を避けるため）。
連続してヘルスチェックに失敗したインスタンスをイベントとして通知し、
UI・監査ログ・将来の外部アラート連携（Slack/メール等）から確認できるようにする。
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

CONSECUTIVE_FAILURES_THRESHOLD = 3


class HealthWatchdog:
    def __init__(
        self,
        *,
        check_interval: float = 30.0,
        failure_threshold: int = CONSECUTIVE_FAILURES_THRESHOLD,
        event_publisher: Optional[Callable[..., Awaitable[dict]]] = None,
        smoke_test_runner: Optional[Callable[[str], Awaitable[dict]]] = None,
    ):
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self.event_publisher = event_publisher
        self.smoke_test_runner = smoke_test_runner
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._failure_counts: dict[str, int] = {}
        self._notified: set[str] = set()
        self._smoke_tested: set[str] = set()

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
                # ウォッチドッグ自体の失敗でアプリを落とさない
                pass
            await asyncio.sleep(self.check_interval)

    async def _check_once(self) -> None:
        from app.server_manager import list_instances

        instances = list_instances()
        running_ids: set[str] = set()

        for item in instances:
            instance_id = str(item.get("instance_id") or "")
            if not instance_id or not item.get("running"):
                continue
            running_ids.add(instance_id)

            if item.get("healthy"):
                if instance_id in self._notified:
                    self._notified.discard(instance_id)
                    await self._publish(
                        "instance_health_recovered",
                        item,
                        message=f"インスタンス {instance_id} のヘルスチェックが回復しました",
                    )
                self._failure_counts.pop(instance_id, None)
                if instance_id not in self._smoke_tested:
                    self._smoke_tested.add(instance_id)
                    await self._run_smoke_test(instance_id)
                continue

            count = self._failure_counts.get(instance_id, 0) + 1
            self._failure_counts[instance_id] = count
            if count >= self.failure_threshold and instance_id not in self._notified:
                self._notified.add(instance_id)
                await self._publish(
                    "instance_unhealthy",
                    item,
                    message=(
                        f"インスタンス {instance_id} が {count} 回連続でヘルスチェックに失敗しています。"
                        " ログ（/api/log）を確認してください。"
                    ),
                )

        # 停止済み/削除済みインスタンスの追跡状態をクリア
        for stale_id in list(self._failure_counts.keys()) + list(self._smoke_tested):
            if stale_id not in running_ids:
                self._failure_counts.pop(stale_id, None)
                self._notified.discard(stale_id)
                self._smoke_tested.discard(stale_id)

    async def _run_smoke_test(self, instance_id: str) -> None:
        if not self.smoke_test_runner:
            return
        try:
            result = await self.smoke_test_runner(instance_id)
        except Exception as exc:
            result = {"instance_id": instance_id, "success": False, "error": str(exc)}
        message = (
            f"起動後の自動疎通テストに成功しました（{result.get('latency_ms')}ms）"
            if result.get("success")
            else f"起動後の自動疎通テストに失敗しました: {result.get('error')}"
        )
        await self._publish("instance_smoke_test", result, message=message)

    async def _publish(self, event_type: str, data: Any, *, message: str) -> None:
        if self.event_publisher:
            await self.event_publisher(event_type, data, message=message, actor="system")
