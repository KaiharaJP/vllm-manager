"""HealthWatchdog の通知ロジックのユニットテスト。"""

import asyncio
import unittest
from unittest.mock import patch

from app.health_watchdog import HealthWatchdog


def _run(coro):
    return asyncio.run(coro)


class HealthWatchdogTests(unittest.TestCase):
    def test_notifies_after_threshold_and_recovers(self):
        watchdog = HealthWatchdog(failure_threshold=2)
        published: list[tuple[str, str]] = []

        async def fake_publisher(event_type, data=None, *, message=None, actor=None):
            published.append((event_type, message or ""))
            return {}

        watchdog.event_publisher = fake_publisher

        unhealthy_item = [{"instance_id": "chat-main", "running": True, "healthy": False}]
        healthy_item = [{"instance_id": "chat-main", "running": True, "healthy": True}]

        with patch("app.server_manager.list_instances", return_value=unhealthy_item):
            _run(watchdog._check_once())
            self.assertEqual(published, [])  # 1回目はまだ閾値未満
            _run(watchdog._check_once())
            self.assertEqual(len(published), 1)
            self.assertEqual(published[0][0], "instance_unhealthy")
            _run(watchdog._check_once())
            self.assertEqual(len(published), 1)  # 既に通知済みなら再通知しない

        with patch("app.server_manager.list_instances", return_value=healthy_item):
            _run(watchdog._check_once())
            self.assertEqual(len(published), 2)
            self.assertEqual(published[1][0], "instance_health_recovered")

    def test_stopped_instance_clears_state(self):
        watchdog = HealthWatchdog(failure_threshold=1)
        published = []

        async def fake_publisher(event_type, data=None, *, message=None, actor=None):
            published.append(event_type)
            return {}

        watchdog.event_publisher = fake_publisher

        with patch(
            "app.server_manager.list_instances",
            return_value=[{"instance_id": "embed-a", "running": True, "healthy": False}],
        ):
            _run(watchdog._check_once())
        self.assertIn("embed-a", watchdog._notified)

        with patch("app.server_manager.list_instances", return_value=[]):
            _run(watchdog._check_once())
        self.assertNotIn("embed-a", watchdog._notified)
        self.assertNotIn("embed-a", watchdog._failure_counts)


if __name__ == "__main__":
    unittest.main()
