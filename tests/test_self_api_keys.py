"""Self-service LiteLLM inference key endpoints (sk- via PAT/JWT)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import app.auth as auth


class SelfApiKeyEndpointTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-self-api-keys-")
        self._prev_data_dir = auth.DATA_DIR
        self._prev_users_file = auth.USERS_FILE
        self._prev_api_keys_file = auth.API_KEYS_FILE

        auth.DATA_DIR = Path(self._tmpdir)
        auth.USERS_FILE = auth.DATA_DIR / "users.json"
        auth.API_KEYS_FILE = auth.DATA_DIR / "api_keys.json"
        auth._sessions.clear()

        self._user = {
            "username": "teacher1",
            "role": "user",
            "litellm_user_id": "teacher1",
            "litellm_team_id": "users",
        }

        import app.main as main_module

        self.main_module = main_module
        main_module.app.dependency_overrides[main_module.get_current_user] = lambda: self._user

        from fastapi.testclient import TestClient

        self.client = TestClient(main_module.app)

    def tearDown(self):
        self.main_module.app.dependency_overrides.clear()
        auth.DATA_DIR = self._prev_data_dir
        auth.USERS_FILE = self._prev_users_file
        auth.API_KEYS_FILE = self._prev_api_keys_file
        auth._sessions.clear()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_create_my_api_key_proxies_to_litellm_generate(self):
        fake = {"key": "sk-test-generated-key", "token_id": "tok-1"}

        with patch(
            "app.main.litellm_request",
            new_callable=AsyncMock,
            return_value=fake,
        ) as mocked:
            resp = self.client.post(
                "/api/auth/me/api-keys",
                json={"models": ["*"]},
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["key"], "sk-test-generated-key")
        mocked.assert_awaited_once()
        args = mocked.await_args
        self.assertEqual(args.args[0], "POST")
        self.assertEqual(args.args[1], "/key/generate")
        payload = args.args[2]
        self.assertEqual(payload["models"], ["*"])
        self.assertEqual(payload["user_id"], "teacher1")
        self.assertEqual(payload["team_id"], "users")

    def test_create_my_api_key_rejects_empty_models(self):
        resp = self.client.post(
            "/api/auth/me/api-keys",
            json={"models": ["  ", ""]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_list_my_api_keys_filters_by_user(self):
        detailed = [
            {"info": {"user_id": "teacher1", "key_alias": "mine"}},
            {"info": {"user_id": "other", "key_alias": "theirs"}},
            {"info": {"team_id": "users", "key_alias": "team"}},
        ]

        with patch(
            "app.main._litellm_list_keys_detailed",
            new_callable=AsyncMock,
            return_value=detailed,
        ):
            resp = self.client.get("/api/auth/me/api-keys")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        aliases = {e["info"]["key_alias"] for e in body["keys"]}
        self.assertEqual(aliases, {"mine", "team"})
        self.assertEqual(body["count"], 2)


if __name__ == "__main__":
    unittest.main()
