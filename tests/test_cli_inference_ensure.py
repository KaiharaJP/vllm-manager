"""CLI inference-key ensure (get-or-create) unit tests."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import app.auth as auth
import app.cli_inference_keys as cli_keys


class EnsureCliInferenceKeyTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-cli-ensure-")
        self._prev_auth_data = auth.DATA_DIR
        self._prev_cli_data = cli_keys.DATA_DIR
        self._prev_cli_file = cli_keys.CLI_KEYS_FILE

        auth.DATA_DIR = Path(self._tmpdir)
        cli_keys.DATA_DIR = Path(self._tmpdir)
        cli_keys.CLI_KEYS_FILE = Path(self._tmpdir) / "cli_inference_keys.json"

        self._user = {
            "username": "admin",
            "role": "admin",
            "litellm_user_id": "admin",
            "litellm_team_id": None,
        }

        import app.main as main_module

        self.main_module = main_module
        main_module.app.dependency_overrides[main_module.get_current_user] = lambda: self._user

        from fastapi.testclient import TestClient

        self.client = TestClient(main_module.app)

    def tearDown(self):
        self.main_module.app.dependency_overrides.clear()
        auth.DATA_DIR = self._prev_auth_data
        cli_keys.DATA_DIR = self._prev_cli_data
        cli_keys.CLI_KEYS_FILE = self._prev_cli_file
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_ensure_creates_then_reuses_same_models(self):
        async def fake_litellm(method, path, payload=None):
            if method == "POST" and path == "/key/generate":
                return {"key": "sk-first-key"}
            if method == "GET" and path == "/key/list":
                return {"keys": []}
            raise AssertionError(f"unexpected litellm call {method} {path}")

        with patch("app.cli_inference_keys.litellm_request", side_effect=fake_litellm):
            first = self.client.post(
                "/api/auth/me/api-keys/ensure",
                json={"models": ["*"], "key_alias": "vllm-cli"},
            )
            second = self.client.post(
                "/api/auth/me/api-keys/ensure",
                json={"models": ["*"], "key_alias": "vllm-cli"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(first.json()["reused"])
        self.assertEqual(first.json()["key"], "sk-first-key")
        self.assertTrue(second.json()["reused"])
        self.assertEqual(second.json()["key"], "sk-first-key")

    def test_ensure_force_mints_new_key(self):
        generate_calls = {"n": 0}

        async def fake_litellm(method, path, payload=None):
            if method == "POST" and path == "/key/generate":
                generate_calls["n"] += 1
                return {"key": f"sk-key-{generate_calls['n']}"}
            if method == "GET" and path == "/key/list":
                return {"keys": []}
            if method == "POST" and path == "/key/delete":
                return {"deleted": True}
            raise AssertionError(f"unexpected litellm call {method} {path}")

        with patch("app.cli_inference_keys.litellm_request", side_effect=fake_litellm):
            first = self.client.post(
                "/api/auth/me/api-keys/ensure",
                json={"models": ["*"], "key_alias": "vllm-cli"},
            )
            forced = self.client.post(
                "/api/auth/me/api-keys/ensure",
                json={"models": ["*"], "key_alias": "vllm-cli", "force": True},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(forced.status_code, 200)
        self.assertEqual(first.json()["key"], "sk-key-1")
        self.assertFalse(forced.json()["reused"])
        self.assertEqual(forced.json()["key"], "sk-key-2")

    def test_ensure_different_models_mints_new_key(self):
        generate_calls = {"n": 0}

        async def fake_litellm(method, path, payload=None):
            if method == "POST" and path == "/key/generate":
                generate_calls["n"] += 1
                return {"key": f"sk-model-{generate_calls['n']}"}
            if method == "GET" and path == "/key/list":
                return {"keys": []}
            if method == "POST" and path == "/key/delete":
                return {"deleted": True}
            raise AssertionError(f"unexpected litellm call {method} {path}")

        with patch("app.cli_inference_keys.litellm_request", side_effect=fake_litellm):
            first = self.client.post(
                "/api/auth/me/api-keys/ensure",
                json={"models": ["*"], "key_alias": "vllm-cli"},
            )
            second = self.client.post(
                "/api/auth/me/api-keys/ensure",
                json={"models": ["vllm-local"], "key_alias": "vllm-cli"},
            )

        self.assertEqual(first.json()["key"], "sk-model-1")
        self.assertFalse(second.json()["reused"])
        self.assertEqual(second.json()["key"], "sk-model-2")
        self.assertEqual(second.json()["models"], ["vllm-local"])


if __name__ == "__main__":
    unittest.main()
