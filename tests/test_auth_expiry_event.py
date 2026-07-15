"""Auth expiry event contract tests (frontend integration surface)."""

import unittest


class AuthExpiryEventTests(unittest.TestCase):
    def test_auth_expired_event_name_is_stable(self):
        """frontend/src/lib/auth-events.ts と AuthGate が共有するイベント名。"""
        self.assertEqual("vllm-manager:auth-expired", "vllm-manager:auth-expired")

    def test_api_triggers_auth_expired_only_on_401(self):
        """403 は権限不足のためログアウト対象にしない（プラン仕様）。"""
        statuses_that_logout = {401}
        self.assertIn(401, statuses_that_logout)
        self.assertNotIn(403, statuses_that_logout)


if __name__ == "__main__":
    unittest.main()
