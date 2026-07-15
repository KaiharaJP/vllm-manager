"""EventBus のロール別フィルタテスト。"""

import unittest

from app.event_bus import EventBus


class EventBusSecurityTests(unittest.TestCase):
    def test_admin_only_events_hidden_from_users(self):
        bus = EventBus()
        admin = {"username": "admin", "role": "admin"}
        user = {"username": "teacher", "role": "user"}
        event = {
            "id": "1",
            "type": "litellm_proxy_request",
            "timestamp": 1.0,
            "data": {"request_summary": "1 msg"},
        }
        bus.history.append(event)
        admin_history = bus.get_history(count=10, user=admin)
        user_history = bus.get_history(count=10, user=user)
        self.assertEqual(len(admin_history), 1)
        self.assertEqual(len(user_history), 0)

    def test_sanitize_event_for_user(self):
        bus = EventBus()
        event = {"type": "server_job", "data": {}}
        filtered = bus.sanitize_event_for_user(event, {"role": "user"})
        self.assertIsNotNone(filtered)


if __name__ == "__main__":
    unittest.main()
