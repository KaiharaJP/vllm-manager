"""管理画面チャット UI 用 API と chat_keys のユニットテスト。"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import app.auth as auth
import app.chat_keys as chat_keys
import app.main as main_module


class ChatKeysTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-chat-keys-test-")
        self._prev_data_dir = auth.DATA_DIR
        self._prev_chat_keys_file = chat_keys.CHAT_KEYS_FILE

        auth.DATA_DIR = Path(self._tmpdir)
        chat_keys.DATA_DIR = Path(self._tmpdir)
        chat_keys.CHAT_KEYS_FILE = chat_keys.DATA_DIR / "chat_keys.json"

    def tearDown(self):
        auth.DATA_DIR = self._prev_data_dir
        chat_keys.DATA_DIR = self._prev_data_dir
        chat_keys.CHAT_KEYS_FILE = self._prev_chat_keys_file
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    async def test_get_or_create_issues_key_once(self):
        user = {"username": "teacher1", "litellm_user_id": "teacher1", "litellm_team_id": "users"}
        calls: list[str] = []

        async def fake_litellm(method, path, payload=None):
            calls.append(path)
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/key/generate")
            self.assertEqual(payload["key_alias"], "chat-ui-teacher1")
            self.assertEqual(payload["models"], ["*"])
            return {"key": "sk-chat-test-key"}

        with patch("app.chat_keys.litellm_request", side_effect=fake_litellm):
            key1 = await chat_keys.get_or_create_chat_key(user)
            key2 = await chat_keys.get_or_create_chat_key(user)

        self.assertEqual(key1, "sk-chat-test-key")
        self.assertEqual(key2, "sk-chat-test-key")
        self.assertEqual(calls, ["/key/generate"])
        stored = json.loads(chat_keys.CHAT_KEYS_FILE.read_text())
        self.assertEqual(stored["teacher1"]["litellm_key"], "sk-chat-test-key")

    async def test_issue_chat_key_recovers_from_duplicate_alias(self):
        user = {"username": "admin", "litellm_user_id": "admin", "litellm_team_id": "admins"}
        calls: list[tuple[str, str]] = []
        generate_attempts = 0

        async def fake_litellm(method, path, payload=None):
            calls.append((method, path))
            if method == "POST" and path == "/key/generate":
                nonlocal generate_attempts
                generate_attempts += 1
                if generate_attempts == 1:
                    req = httpx.Request("POST", "http://litellm/key/generate")
                    resp = httpx.Response(
                        400,
                        request=req,
                        json={
                            "error": {
                                "message": "Key with alias 'chat-ui-admin' already exists.",
                            }
                        },
                    )
                    raise httpx.HTTPStatusError("duplicate", request=req, response=resp)
                return {"key": "sk-chat-new"}
            if method == "GET" and path == "/key/list":
                return {"keys": ["hash-admin"]}
            if method == "GET" and path.startswith("/key/info"):
                return {"info": {"key_alias": "chat-ui-admin"}}
            if method == "POST" and path == "/key/delete":
                return {"deleted_keys": ["hash-admin"]}
            raise AssertionError((method, path, payload))

        with patch("app.chat_keys.litellm_request", side_effect=fake_litellm):
            key = await chat_keys._issue_chat_key(user)

        self.assertEqual(key, "sk-chat-new")
        self.assertIn(("POST", "/key/delete"), calls)

    async def test_get_or_create_upgrades_legacy_key_scope(self):
        user = {"username": "teacher1", "litellm_user_id": "teacher1"}
        chat_keys.CHAT_KEYS_FILE.write_text(
            json.dumps(
                {
                    "teacher1": {
                        "litellm_key": "sk-legacy",
                        "key_alias": "chat-ui-teacher1",
                        "models": ["vllm-local"],
                        "created_at": 1.0,
                    }
                }
            )
        )
        calls: list[str] = []

        async def fake_litellm(method, path, payload=None):
            calls.append(path)
            if path == "/key/list":
                return {"keys": []}
            return {"key": "sk-upgraded"}

        with patch("app.chat_keys.litellm_request", side_effect=fake_litellm):
            key = await chat_keys.get_or_create_chat_key(user)

        self.assertEqual(key, "sk-upgraded")
        self.assertIn("/key/generate", calls)
        stored = json.loads(chat_keys.CHAT_KEYS_FILE.read_text())
        self.assertEqual(stored["teacher1"]["models"], ["*"])

    async def test_regenerate_replaces_stored_key(self):
        user = {"username": "teacher1", "litellm_user_id": "teacher1"}
        seq = iter(["sk-old", "sk-new"])

        async def fake_litellm(method, path, payload=None):
            if path == "/key/list":
                return {"keys": []}
            return {"key": next(seq)}

        with patch("app.chat_keys.litellm_request", side_effect=fake_litellm):
            old = await chat_keys.get_or_create_chat_key(user)
            new = await chat_keys.regenerate_chat_key(user)

        self.assertEqual(old, "sk-old")
        self.assertEqual(new, "sk-new")
        stored = json.loads(chat_keys.CHAT_KEYS_FILE.read_text())
        self.assertEqual(stored["teacher1"]["litellm_key"], "sk-new")


class ChatEndpointTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-chat-api-test-")
        self._prev_data_dir = auth.DATA_DIR
        self._prev_chat_keys_file = chat_keys.CHAT_KEYS_FILE

        auth.DATA_DIR = Path(self._tmpdir)
        chat_keys.DATA_DIR = Path(self._tmpdir)
        chat_keys.CHAT_KEYS_FILE = chat_keys.DATA_DIR / "chat_keys.json"

        self._user = {
            "username": "teacher1",
            "role": "user",
            "litellm_user_id": "teacher1",
            "litellm_team_id": "users",
        }
        main_module.app.dependency_overrides[main_module.get_current_user] = lambda: self._user

        from fastapi.testclient import TestClient

        self.client = TestClient(main_module.app)

    def tearDown(self):
        main_module.app.dependency_overrides.clear()
        auth.DATA_DIR = self._prev_data_dir
        chat_keys.DATA_DIR = self._prev_data_dir
        chat_keys.CHAT_KEYS_FILE = self._prev_chat_keys_file
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_chat_models_returns_running_model_ids(self):
        fake_models = [{"id": "Qwen/Qwen3.6-27B-FP8", "object": "model"}]

        with (
            patch("app.main.get_or_create_chat_key", new_callable=AsyncMock, return_value="sk-chat"),
            patch("app.main._allowed_models_from_litellm_key", new_callable=AsyncMock, return_value=["*"]),
            patch(
                "app.main.list_running_servers",
                return_value=[
                    {
                        "port": 8011,
                        "model": "Qwen/Qwen3.6-27B-FP8",
                        "task_type": "chat",
                    }
                ],
            ),
            patch("app.main.get_status", return_value={"running": True, "vllm_port": 8011}),
            patch("app.main._collect_active_vllm_models", new_callable=AsyncMock, return_value=fake_models),
        ):
            resp = self.client.get("/api/chat/models")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["models"], ["Qwen/Qwen3.6-27B-FP8"])
        self.assertEqual(body["count"], 1)

    def test_chat_models_excludes_embedding_servers(self):
        with (
            patch("app.main.get_or_create_chat_key", new_callable=AsyncMock, return_value="sk-chat"),
            patch("app.main._allowed_models_from_litellm_key", new_callable=AsyncMock, return_value=["*"]),
            patch(
                "app.main.list_running_servers",
                return_value=[
                    {"port": 8012, "model": "intfloat/e5-large", "task_type": "embedding"},
                ],
            ),
            patch("app.main.get_status", return_value={"running": False}),
            patch("app.main._collect_active_vllm_models", new_callable=AsyncMock, return_value=[]),
        ):
            resp = self.client.get("/api/chat/models")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["models"], [])

    def test_chat_models_empty_when_no_servers(self):
        with (
            patch("app.main.get_or_create_chat_key", new_callable=AsyncMock, return_value="sk-chat"),
            patch("app.main._allowed_models_from_litellm_key", new_callable=AsyncMock, return_value=["vllm-local"]),
            patch("app.main.list_running_servers", return_value=[]),
            patch("app.main.get_status", return_value={"running": False}),
        ):
            resp = self.client.get("/api/chat/models")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["models"], [])

    def test_chat_completions_non_streaming_proxy(self):
        captured: dict = {}

        async def fake_proxy(*, chat_key, payload, stream):
            captured["chat_key"] = chat_key
            captured["payload"] = payload
            captured["stream"] = stream
            return main_module.Response(
                content=json.dumps(
                    {
                        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                    }
                ).encode("utf-8"),
                status_code=200,
                media_type="application/json",
            )

        with (
            patch("app.main.get_or_create_chat_key", new_callable=AsyncMock, return_value="sk-chat"),
            patch("app.main._proxy_chat_to_litellm", side_effect=fake_proxy),
        ):
            resp = self.client.post(
                "/api/chat/completions",
                json={
                    "model": "vllm-local",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(captured["chat_key"], "sk-chat")
        self.assertFalse(captured["stream"])
        self.assertEqual(captured["payload"]["model"], "vllm-local")
        self.assertEqual(resp.json()["choices"][0]["message"]["content"], "hello")

    def test_chat_completions_retries_on_401(self):
        attempts = {"count": 0}

        async def fake_proxy(*, chat_key, payload, stream):
            attempts["count"] += 1
            if attempts["count"] == 1:
                return main_module.Response(content=b'{"error":"unauthorized"}', status_code=401)
            return main_module.Response(
                content=json.dumps(
                    {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
                ).encode("utf-8"),
                status_code=200,
                media_type="application/json",
            )

        with (
            patch("app.main.get_or_create_chat_key", new_callable=AsyncMock, return_value="sk-old"),
            patch("app.main.regenerate_chat_key", new_callable=AsyncMock, return_value="sk-new") as regen,
            patch("app.main._proxy_chat_to_litellm", side_effect=fake_proxy),
        ):
            resp = self.client.post(
                "/api/chat/completions",
                json={
                    "model": "vllm-local",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )

        self.assertEqual(resp.status_code, 200)
        regen.assert_awaited_once()
        self.assertEqual(attempts["count"], 2)
        self.assertEqual(resp.json()["choices"][0]["message"]["content"], "ok")

    def test_chat_completions_streaming_returns_event_stream(self):
        async def fake_proxy(*, chat_key, payload, stream):
            self.assertTrue(stream)

            async def gen():
                yield b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
                yield b"data: [DONE]\n\n"

            return main_module.StreamingResponse(gen(), media_type="text/event-stream")

        with (
            patch("app.main.get_or_create_chat_key", new_callable=AsyncMock, return_value="sk-chat"),
            patch("app.main._proxy_chat_to_litellm", side_effect=fake_proxy),
        ):
            resp = self.client.post(
                "/api/chat/completions",
                json={
                    "model": "vllm-local",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
        self.assertIn(b"Hi", resp.content)


class ProxyChatToLiteLLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_non_stream_posts_to_litellm(self):
        request = MagicMock()
        request.method = "POST"
        request.url = "http://litellm:4000/v1/chat/completions"

        response = MagicMock()
        response.status_code = 200
        response.content = b'{"ok":true}'
        response.headers = {"content-type": "application/json"}

        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.main.httpx.AsyncClient", return_value=client):
            result = await main_module._proxy_chat_to_litellm(
                chat_key="sk-test",
                payload={"model": "vllm-local", "messages": [], "stream": False},
                stream=False,
            )

        self.assertEqual(result.status_code, 200)
        client.post.assert_awaited_once()
        args, kwargs = client.post.call_args
        self.assertTrue(args[0].endswith("/v1/chat/completions"))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-test")


if __name__ == "__main__":
    unittest.main()
