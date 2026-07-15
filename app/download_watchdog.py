"""モデルダウンロードジョブの進捗停止を検知して自動再開する。"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable, Optional

DEFAULT_CHECK_INTERVAL_SEC = float(os.environ.get("DOWNLOAD_WATCHDOG_INTERVAL_SEC", "30"))


class DownloadWatchdog:
    def __init__(
        self,
        *,
        check_interval: float = DEFAULT_CHECK_INTERVAL_SEC,
        inspector: Optional[Callable[[], Awaitable[list[dict[str, Any]]]]] = None,
    ):
        self.check_interval = check_interval
        self.inspector = inspector
        self._running = False
        self._task: Optional[asyncio.Task] = None

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

    async def _check_once(self) -> None:
        if self.inspector:
            await self.inspector()
