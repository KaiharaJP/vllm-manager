"""管理者による他ユーザーの永続APIトークン（PAT）管理のユニットテスト。

注意: ここでは `importlib.reload(app.auth)` を使わない。reload は
`require_admin` 等の関数オブジェクトを再生成してしまい、`app.main` の
ルートに埋め込まれた `Depends(require_admin)` の実体と食い違うことで、
本テストファイル以降に実行される他のテスト（`TestClient` +
`dependency_overrides` を使うもの）を壊す可能性があるため、モジュール
属性（DATA_DIR 等）を直接書き換える方式で隔離する。
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import app.auth as auth


class AdminTokenEndpointTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-admin-token-test-")
        self._prev_data_dir = auth.DATA_DIR
        self._prev_users_file = auth.USERS_FILE
        self._prev_api_keys_file = auth.API_KEYS_FILE

        auth.DATA_DIR = Path(self._tmpdir)
        auth.USERS_FILE = auth.DATA_DIR / "users.json"
        auth.API_KEYS_FILE = auth.DATA_DIR / "api_keys.json"
        auth._sessions.clear()

        from fastapi.testclient import TestClient
        import app.main as main_module

        self.main_module = main_module
        main_module.app.dependency_overrides[main_module.require_admin] = lambda: {
            "username": "admin",
            "role": "admin",
        }
        self.client = TestClient(main_module.app)

        auth.upsert_user("teacher1", "strong-password-here", role="user")

    def tearDown(self):
        self.main_module.app.dependency_overrides.clear()
        auth.DATA_DIR = self._prev_data_dir
        auth.USERS_FILE = self._prev_users_file
        auth.API_KEYS_FILE = self._prev_api_keys_file
        auth._sessions.clear()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_admin_can_list_other_users_tokens(self):
        public_record, _raw_key = auth.create_api_key("teacher1", "automation")

        resp = self.client.get("/api/users/teacher1/tokens")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["tokens"][0]["id"], public_record["id"])
        self.assertNotIn("key_hash", body["tokens"][0])
        self.assertNotIn("token", body["tokens"][0])

    def test_admin_can_force_revoke_other_users_token(self):
        public_record, raw_key = auth.create_api_key("teacher1", "automation")

        resp = self.client.delete(f"/api/users/teacher1/tokens/{public_record['id']}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            auth.get_user_from_api_key(raw_key)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_listing_tokens_for_unknown_user_returns_404(self):
        resp = self.client.get("/api/users/ghost/tokens")
        self.assertEqual(resp.status_code, 404)

    def test_revoking_nonexistent_token_returns_404(self):
        resp = self.client.delete("/api/users/teacher1/tokens/does-not-exist")
        self.assertEqual(resp.status_code, 404)

    def test_admin_cannot_revoke_token_via_wrong_username(self):
        """token_id は正しいが username が違う場合は失効させない（他人の資源保護）。"""
        auth.upsert_user("teacher2", "another-strong-password", role="user")
        public_record, raw_key = auth.create_api_key("teacher1", "automation")

        resp = self.client.delete(f"/api/users/teacher2/tokens/{public_record['id']}")
        self.assertEqual(resp.status_code, 404)

        # まだ有効なはず
        user = auth.get_user_from_api_key(raw_key)
        self.assertEqual(user["username"], "teacher1")


if __name__ == "__main__":
    unittest.main()
