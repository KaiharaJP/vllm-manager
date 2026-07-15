"""ResourceWatchdog（GPU温度・VRAM・ディスクしきい値監視）のユニットテスト。"""

import asyncio
import unittest

from app.resource_watchdog import ResourceWatchdog


def _run(coro):
    return asyncio.run(coro)


def _metrics(gpu_temp=70.0, gpu_mem_used=1000.0, gpu_mem_total=10000.0, disk_percent=50.0):
    return {
        "gpus": [
            {
                "index": 0,
                "temperature_c": gpu_temp,
                "memory_used_mb": gpu_mem_used,
                "memory_total_mb": gpu_mem_total,
            }
        ],
        "disks": [{"label": "root", "path": "/", "usage_percent": disk_percent}],
    }


class ResourceWatchdogTests(unittest.TestCase):
    def _watchdog(self, published, **overrides):
        async def fake_publisher(event_type, data=None, *, message=None, actor=None):
            published.append((event_type, data.get("resource") if data else None))
            return {}

        return ResourceWatchdog(
            event_publisher=fake_publisher,
            gpu_temp_threshold_c=overrides.get("gpu_temp_threshold_c", 85.0),
            gpu_memory_threshold_percent=overrides.get("gpu_memory_threshold_percent", 95.0),
            disk_threshold_percent=overrides.get("disk_threshold_percent", 90.0),
        )

    def test_no_alert_below_threshold(self):
        published: list[tuple[str, str]] = []
        watchdog = self._watchdog(published)
        watchdog.metrics_provider = lambda: _metrics(gpu_temp=70.0, disk_percent=50.0)
        _run(watchdog._check_once())
        self.assertEqual(published, [])

    def test_gpu_temp_alert_and_recovery(self):
        published: list[tuple[str, str]] = []
        watchdog = self._watchdog(published, gpu_temp_threshold_c=85.0)

        watchdog.metrics_provider = lambda: _metrics(gpu_temp=90.0)
        _run(watchdog._check_once())
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0][0], "resource_alert")
        self.assertEqual(published[0][1], "gpu:0:temp")

        # 閾値超過が続く間は再通知しない
        _run(watchdog._check_once())
        self.assertEqual(len(published), 1)

        # 閾値未満でもヒステリシス内（threshold - margin より高い）なら回復通知しない
        watchdog.metrics_provider = lambda: _metrics(gpu_temp=82.0)
        _run(watchdog._check_once())
        self.assertEqual(len(published), 1)

        # ヒステリシスを超えて下がったら回復通知
        watchdog.metrics_provider = lambda: _metrics(gpu_temp=70.0)
        _run(watchdog._check_once())
        self.assertEqual(len(published), 2)
        self.assertEqual(published[1][0], "resource_alert_recovered")

    def test_gpu_memory_alert(self):
        published: list[tuple[str, str]] = []
        watchdog = self._watchdog(published, gpu_memory_threshold_percent=90.0)
        watchdog.metrics_provider = lambda: _metrics(gpu_mem_used=9500.0, gpu_mem_total=10000.0)
        _run(watchdog._check_once())
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0][1], "gpu:0:memory")

    def test_disk_alert(self):
        published: list[tuple[str, str]] = []
        watchdog = self._watchdog(published, disk_threshold_percent=90.0)
        watchdog.metrics_provider = lambda: _metrics(disk_percent=95.0)
        _run(watchdog._check_once())
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0][1], "disk:root")

    def test_stale_resource_state_is_cleared(self):
        published: list[tuple[str, str]] = []
        watchdog = self._watchdog(published)
        watchdog.metrics_provider = lambda: _metrics(gpu_temp=90.0)
        _run(watchdog._check_once())
        self.assertIn("gpu:0:temp", watchdog._notified)

        # GPU がなくなった（インスタンス変更等）場合は追跡状態をクリア
        watchdog.metrics_provider = lambda: {"gpus": [], "disks": []}
        _run(watchdog._check_once())
        self.assertNotIn("gpu:0:temp", watchdog._notified)


if __name__ == "__main__":
    unittest.main()
