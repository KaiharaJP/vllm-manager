"""認証・セッション・レート制限のユニットテスト。"""

import json
import shutil
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from fastapi import HTTPException

import app.auth as auth


class AuthSecurityTests(unittest.TestCase):
    """モジュール属性を直接差し替えて状態を隔離する。

    以前は `importlib.reload(app.auth)` を使っていたが、reload は
    `require_admin` 等の関数オブジェクトを再生成してしまい、`app.main` の
    ルートに埋め込まれた `Depends(require_admin)` の実体と食い違って
    しまう（他のテストファイルで `TestClient` + `dependency_overrides`
    を使う際に 401 になる原因になる）。そのため属性の直接差し替えに
    統一している。
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-auth-test-")
        self.auth = auth
        self._prev_data_dir = auth.DATA_DIR
        self._prev_users_file = auth.USERS_FILE
        self._prev_api_keys_file = auth.API_KEYS_FILE
        self._prev_ttl = auth.SESSION_TTL_SEC
        self._prev_max_attempts = auth.LOGIN_MAX_ATTEMPTS

        auth.DATA_DIR = Path(self._tmpdir)
        auth.USERS_FILE = auth.DATA_DIR / "users.json"
        auth.API_KEYS_FILE = auth.DATA_DIR / "api_keys.json"
        auth.SESSION_TTL_SEC = 3600
        auth.LOGIN_MAX_ATTEMPTS = 3
        auth._sessions.clear()
        auth._login_failures.clear()
        auth._weak_password_cache.clear()

    def tearDown(self):
        auth.DATA_DIR = self._prev_data_dir
        auth.USERS_FILE = self._prev_users_file
        auth.API_KEYS_FILE = self._prev_api_keys_file
        auth.SESSION_TTL_SEC = self._prev_ttl
        auth.LOGIN_MAX_ATTEMPTS = self._prev_max_attempts
        auth._sessions.clear()
        auth._login_failures.clear()
        auth._weak_password_cache.clear()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_session_expires(self):
        user = {"username": "alice", "role": "user", "litellm_user_id": "alice"}
        token = self.auth.create_session(user)
        session = self.auth._sessions[token]
        session["expires_at"] = time.time() - 1
        with self.assertRaises(HTTPException) as ctx:
            self.auth.get_user_from_token(token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("expired", str(ctx.exception.detail).lower())

    def test_login_rate_limit(self):
        for _ in range(3):
            self.auth.record_login_failure("bob", "127.0.0.1")
        with self.assertRaises(HTTPException) as ctx:
            self.auth.check_login_allowed("bob", "127.0.0.1")
        self.assertEqual(ctx.exception.status_code, 429)
        self.auth.record_login_success("bob", "127.0.0.1")
        self.auth.check_login_allowed("bob", "127.0.0.1")

    def test_weak_password_forces_change_for_any_role(self):
        # デフォルト管理者アカウント（admin/admin）は既知の弱いパスワード。
        self.assertTrue(self.auth.user_must_change_password("admin"))
        self.assertIn("admin", self.auth.find_weak_password_users())
        self.assertIn("admin", self.auth.find_weak_admin_accounts())

        self.auth.upsert_user("teacher1", "correct horse battery staple", role="user")
        self.assertFalse(self.auth.user_must_change_password("teacher1"))

        self.auth.upsert_user("teacher2", "password", role="user")
        self.assertTrue(self.auth.user_must_change_password("teacher2"))
        self.assertIn("teacher2", self.auth.find_weak_password_users())
        # 一般ユーザーは管理者専用の一覧には含まれない
        self.assertNotIn("teacher2", self.auth.find_weak_admin_accounts())

    def test_password_change_clears_forced_change_flag(self):
        self.auth.upsert_user("teacher3", "changeme", role="user")
        self.assertTrue(self.auth.user_must_change_password("teacher3"))
        self.auth.update_user("teacher3", password="a-much-stronger-passphrase")
        self.assertFalse(self.auth.user_must_change_password("teacher3"))

    def test_api_key_created_and_verified(self):
        self.auth.upsert_user("cliuser", "strong-password-here", role="admin")
        public_record, raw_key = self.auth.create_api_key("cliuser", "automation")
        self.assertTrue(raw_key.startswith(self.auth.API_KEY_PREFIX))
        self.assertEqual(public_record["name"], "automation")
        self.assertEqual(public_record["username"], "cliuser")

        user = self.auth.get_user_from_api_key(raw_key)
        self.assertEqual(user["username"], "cliuser")
        self.assertEqual(user["role"], "admin")

    def test_api_key_hash_not_reversible(self):
        self.auth.upsert_user("cliuser2", "strong-password-here", role="user")
        _, raw_key = self.auth.create_api_key("cliuser2", "test-key")
        stored = self.auth.load_api_keys()
        for record in stored.values():
            self.assertNotIn(raw_key, json.dumps(record))
            self.assertEqual(record["key_hash"], self.auth._hash_api_key(raw_key))

    def test_api_key_revocation(self):
        self.auth.upsert_user("cliuser3", "strong-password-here", role="admin")
        public_record, raw_key = self.auth.create_api_key("cliuser3", "revoke-me")
        self.assertTrue(self.auth.revoke_api_key(public_record["id"], username="cliuser3"))
        with self.assertRaises(HTTPException) as ctx:
            self.auth.get_user_from_api_key(raw_key)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_api_key_expiry(self):
        self.auth.upsert_user("cliuser4", "strong-password-here", role="admin")
        public_record, raw_key = self.auth.create_api_key("cliuser4", "expiring", expires_in_days=1)
        keys = self.auth.load_api_keys()
        keys[public_record["id"]]["expires_at"] = time.time() - 1
        self.auth.save_api_keys(keys)
        with self.assertRaises(HTTPException) as ctx:
            self.auth.get_user_from_api_key(raw_key)
        self.assertIn("expired", str(ctx.exception.detail).lower())

    def test_get_user_from_token_dispatches_by_prefix(self):
        session_user = {"username": "sessionuser", "role": "admin", "litellm_user_id": "sessionuser"}
        session_token = self.auth.create_session(session_user)

        self.auth.upsert_user("patuser", "strong-password-here", role="admin")
        _, raw_key = self.auth.create_api_key("patuser", "pat")

        self.assertEqual(self.auth.get_user_from_token(session_token)["username"], "sessionuser")
        self.assertEqual(self.auth.get_user_from_token(raw_key)["username"], "patuser")

    def test_authenticate_seeds_weak_password_cache(self):
        self.auth.upsert_user("cacheuser", "password", role="user")
        self.auth._weak_password_cache.clear()

        with unittest.mock.patch.object(self.auth, "verify_password", wraps=self.auth.verify_password) as verify_mock:
            user = self.auth.authenticate("cacheuser", "password")
            self.assertIsNotNone(user)
            self.assertEqual(verify_mock.call_count, 1)

            self.assertTrue(self.auth.user_must_change_password("cacheuser"))
            self.assertEqual(verify_mock.call_count, 1)

    def test_weak_password_cache_reused_without_bcrypt_rescan(self):
        self.auth.upsert_user("cachedweak", "admin", role="admin")
        password_hash = self.auth.load_users()["cachedweak"]["password_hash"]
        self.assertTrue(self.auth._weak_password_cache.get(password_hash))

        with unittest.mock.patch.object(self.auth, "verify_password", wraps=self.auth.verify_password) as verify_mock:
            self.assertTrue(self.auth.user_must_change_password("cachedweak"))
            self.assertIn("cachedweak", self.auth.find_weak_admin_accounts())
            verify_mock.assert_not_called()

    def test_password_change_updates_weak_password_cache(self):
        self.auth.upsert_user("cacheflip", "changeme", role="user")
        old_hash = self.auth.load_users()["cacheflip"]["password_hash"]
        self.assertTrue(self.auth._weak_password_cache.get(old_hash))

        self.auth.update_user("cacheflip", password="a-much-stronger-passphrase")
        new_hash = self.auth.load_users()["cacheflip"]["password_hash"]
        self.assertFalse(self.auth._weak_password_cache.get(new_hash))
        self.assertFalse(self.auth.user_must_change_password("cacheflip"))


if __name__ == "__main__":
    unittest.main()
