"""複数 vLLM インスタンス・embedding 対応のユニットテスト。"""

import importlib
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

# 各テスト前に server_manager の DATA_DIR を隔離する
_TEST_DATA_ENV = "VLLM_MANAGER_DATA_DIR"


def _reload_server_manager():
    import app.server_manager as sm

    importlib.reload(sm)
    return sm


class ServerManagerInstanceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-mgr-test-")
        self._prev = os.environ.get(_TEST_DATA_ENV)
        os.environ[_TEST_DATA_ENV] = self._tmpdir
        self.sm = _reload_server_manager()

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(_TEST_DATA_ENV, None)
        else:
            os.environ[_TEST_DATA_ENV] = self._prev
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_normalize_task_type(self):
        self.assertEqual(self.sm._normalize_task_type("embedding"), "embedding")
        self.assertEqual(self.sm._normalize_task_type("rerank"), "rerank")
        self.assertEqual(self.sm._normalize_task_type("CHAT"), "chat")
        self.assertEqual(self.sm._normalize_task_type("unknown"), "chat")
        self.assertEqual(self.sm._normalize_task_type(None), "chat")

    def test_sanitize_instance_id(self):
        self.assertEqual(self.sm._sanitize_instance_id("My Instance!"), "My-Instance")
        self.assertEqual(self.sm._sanitize_instance_id("  "), "")

    def test_resolve_new_instance_id(self):
        self.assertEqual(
            self.sm._resolve_new_instance_id(
                instance_id="embed-1", instance_name=None, model_id="x/y"
            ),
            "embed-1",
        )
        self.assertEqual(
            self.sm._resolve_new_instance_id(
                instance_id=None, instance_name="chat main", model_id="x/y"
            ),
            "chat-main",
        )
        generated = self.sm._resolve_new_instance_id(
            instance_id=None, instance_name=None, model_id="jinaai/jina-embeddings-v3"
        )
        self.assertTrue(generated.startswith("inst-jina-embeddings-v3-"))

    def test_instance_config_isolation(self):
        default_cfg = self.sm.load_config()
        default_cfg["model_id"] = "default-model"
        self.sm.save_config(default_cfg)

        embed_cfg = self.sm.load_config("embed-a")
        embed_cfg["model_id"] = "jinaai/jina-embeddings-v3"
        embed_cfg["task_type"] = "embedding"
        embed_cfg["vllm_port"] = 8012
        self.sm.save_config(embed_cfg, instance_id="embed-a")

        self.assertEqual(self.sm.load_config()["model_id"], "default-model")
        loaded = self.sm.load_config("embed-a")
        self.assertEqual(loaded["model_id"], "jinaai/jina-embeddings-v3")
        self.assertEqual(loaded["task_type"], "embedding")
        self.assertEqual(loaded["vllm_port"], 8012)

        paths = self.sm._instance_paths("embed-a")
        self.assertTrue(paths["config"].exists())
        self.assertNotEqual(paths["config"], self.sm.CONFIG_FILE)

    def test_registry_upsert_and_list(self):
        self.sm._upsert_instance_registry(
            {
                "instance_id": "chat-a",
                "instance_name": "Chat A",
                "task_type": "chat",
            }
        )
        self.sm._upsert_instance_registry(
            {
                "instance_id": "embed-a",
                "instance_name": "Embed A",
                "task_type": "embedding",
            }
        )
        instances = self.sm.list_instances()
        ids = {item["instance_id"] for item in instances}
        self.assertIn("chat-a", ids)
        self.assertIn("embed-a", ids)
        embed = next(i for i in instances if i["instance_id"] == "embed-a")
        self.assertEqual(embed["task_type"], "embedding")

    def test_build_command_embedding_vs_chat(self):
        base = {
            "model_id": "jinaai/jina-embeddings-v3",
            "revision": None,
            "vllm_port": 8012,
            "context_length": 8192,
            "gpu_memory_utilization": 0.5,
            "tensor_parallel_size": 1,
            "max_num_seqs": 4,
            "trust_remote_code": False,
            "speculative_config": None,
            "enable_auto_tool_choice": False,
            "tool_call_parser": "",
            "limit_mm_per_prompt": None,
            "mm_encoder_tp_mode": "",
            "mm_processor_cache_type": "",
        }
        embed_cmd = self.sm._build_command({**base, "task_type": "embedding"})
        self.assertIn("--runner", embed_cmd)
        self.assertIn("pooling", embed_cmd)
        self.assertNotIn("--max-num-seqs", embed_cmd)
        self.assertNotIn("--enable-auto-tool-choice", embed_cmd)

        rerank_cmd = self.sm._build_command(
            {**base, "task_type": "rerank", "model_id": "BAAI/bge-reranker-v2-m3"}
        )
        self.assertIn("--runner", rerank_cmd)
        self.assertIn("pooling", rerank_cmd)
        self.assertNotIn("--max-num-seqs", rerank_cmd)

        chat_cmd = self.sm._build_command(
            {
                **base,
                "task_type": "chat",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "enable_auto_tool_choice": True,
                "tool_call_parser": "hermes",
            }
        )
        self.assertIn("--max-num-seqs", chat_cmd)
        self.assertNotIn("--runner", chat_cmd)
        self.assertIn("--enable-auto-tool-choice", chat_cmd)

    def test_default_models_include_embedding(self):
        models = self.sm.get_available_models()
        embed = [m for m in models if m.get("task_type") == "embedding"]
        jina = next((m for m in embed if m["id"] == "jinaai/jina-embeddings-v3"), None)
        self.assertIsNotNone(jina)
        self.assertEqual(jina["recommended_context_length"], 8192)

    def test_build_command_embedding_model(self):
        base = {
            "model_id": "org/custom-embedding",
            "revision": None,
            "vllm_port": 8013,
            "context_length": 8192,
            "gpu_memory_utilization": 0.5,
            "tensor_parallel_size": 1,
            "max_num_seqs": 4,
            "trust_remote_code": True,
            "speculative_config": {"method": "ngram"},
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
            "limit_mm_per_prompt": {"image": 1},
            "mm_encoder_tp_mode": "",
            "mm_processor_cache_type": "",
        }
        cmd = self.sm._build_command({**base, "task_type": "embedding"})
        self.assertIn("--runner", cmd)
        self.assertIn("pooling", cmd)
        self.assertIn("--max-model-len", cmd)
        self.assertIn("8192", cmd)
        self.assertIn("--trust-remote-code", cmd)
        self.assertNotIn("--max-num-seqs", cmd)
        self.assertNotIn("--enable-auto-tool-choice", cmd)
        self.assertNotIn("--speculative-config", cmd)


class ProxyRoutingTests(unittest.TestCase):
    def setUp(self):
        from app.main import (
            _pick_target_server,
            _preferred_task_type_for_subpath,
        )

        self.pick = _pick_target_server
        self.subpath_type = _preferred_task_type_for_subpath
        self.servers = [
            {
                "port": 8001,
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "task_type": "chat",
                "managed_by_app": True,
            },
            {
                "port": 8012,
                "model": "jinaai/jina-embeddings-v3",
                "task_type": "embedding",
                "managed_by_app": True,
            },
        ]

    def test_preferred_task_type_for_subpath(self):
        self.assertEqual(self.subpath_type("embeddings"), "embedding")
        self.assertEqual(self.subpath_type("chat/completions"), "chat")
        self.assertEqual(self.subpath_type("score"), "rerank")
        self.assertEqual(self.subpath_type("rerank"), "rerank")
        self.assertEqual(self.subpath_type("v1/rerank"), "rerank")
        self.assertEqual(self.subpath_type("v2/rerank"), "rerank")
        self.assertIsNone(self.subpath_type("models"))

    def test_pick_embedding_server(self):
        target = self.pick(
            "jinaai/jina-embeddings-v3",
            self.servers,
            preferred_task_type="embedding",
        )
        self.assertEqual(target["port"], 8012)

    def test_pick_chat_server(self):
        target = self.pick(
            "Qwen/Qwen2.5-7B-Instruct",
            self.servers,
            preferred_task_type="chat",
        )
        self.assertEqual(target["port"], 8001)

    def test_embedding_request_skips_chat_only_on_chat_server(self):
        target = self.pick(
            "jinaai/jina-embeddings-v3",
            self.servers,
            preferred_task_type="embedding",
        )
        self.assertEqual(target["task_type"], "embedding")
        chat_only = [self.servers[0]]
        self.assertIsNone(
            self.pick(
                "jinaai/jina-embeddings-v3",
                chat_only,
                preferred_task_type="embedding",
            )
        )

    def test_alias_model_picks_typed_server(self):
        target = self.pick("vllm-local", self.servers, preferred_task_type="embedding")
        self.assertEqual(target["port"], 8012)
        target_chat = self.pick("vllm-local", self.servers, preferred_task_type="chat")
        self.assertEqual(target_chat["port"], 8001)


class ApiInstancesEndpointTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-mgr-api-test-")
        self._prev = os.environ.get(_TEST_DATA_ENV)
        os.environ[_TEST_DATA_ENV] = self._tmpdir
        self.sm = _reload_server_manager()
        import app.model_manager as mm

        importlib.reload(mm)
        self.mm = mm
        import app.main as main_module

        importlib.reload(main_module)

        from fastapi.testclient import TestClient
        from app.main import app
        from app.auth import require_admin

        app.dependency_overrides[require_admin] = lambda: {"username": "admin", "role": "admin"}
        self.client = TestClient(app)

    def tearDown(self):
        import app.main as main_module

        main_module.app.dependency_overrides.clear()
        if self._prev is None:
            os.environ.pop(_TEST_DATA_ENV, None)
        else:
            os.environ[_TEST_DATA_ENV] = self._prev
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_instances_endpoint_empty(self):
        resp = self.client.get("/api/instances")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_instances_endpoint_with_registry(self):
        self.sm._upsert_instance_registry(
            {
                "instance_id": "test-embed",
                "instance_name": "Test Embed",
                "task_type": "embedding",
            }
        )
        resp = self.client.get("/api/instances")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["instance_id"], "test-embed")
        self.assertEqual(data[0]["task_type"], "embedding")

    def test_start_request_validates_task_type(self):
        resp = self.client.post(
            "/api/start",
            json={"model_id": "jinaai/jina-embeddings-v3", "task_type": "invalid"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_start_resolves_embedding_from_catalog(self):
        from unittest.mock import patch

        catalog_entry = {
            "id": "jinaai/jina-embeddings-v3",
            "task_type": "embedding",
            "recommended_context_length": 8192,
            "trust_remote_code": False,
        }
        with patch("app.main.load_model_catalog", return_value=[catalog_entry]):
            with patch("app.main.start_server", return_value={"success": True, "message": "ok", "steps": []}) as mock_start:
                resp = self.client.post(
                    "/api/start",
                    json={"model_id": "jinaai/jina-embeddings-v3"},
                )
        self.assertEqual(resp.status_code, 200, resp.text)
        kwargs = mock_start.call_args.kwargs
        self.assertEqual(kwargs["task_type"], "embedding")
        self.assertEqual(kwargs["context_length"], 8192)
        self.assertTrue(kwargs["create_new_instance"])

    def test_model_download_requires_registration(self):
        resp = self.client.post(
            "/api/model-downloads",
            json={"model_id": "org/unregistered-embedding"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not registered", resp.json().get("detail", ""))

    def test_model_download_accepts_registered_embedding(self):
        from unittest.mock import patch

        self.mm.save_model(
            {
                "id": "org/custom-embedding",
                "name": "Custom Embedding",
                "task_type": "embedding",
                "recommended_context_length": 8192,
            }
        )
        with patch("asyncio.create_task"):
            resp = self.client.post(
                "/api/model-downloads",
                json={"model_id": "org/custom-embedding"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["model_id"], "org/custom-embedding")
        self.assertEqual(resp.json()["status"], "queued")

    def test_embeddings_returns_503_when_only_chat_server_running(self):
        from unittest.mock import patch

        chat_server = {
            "port": 8001,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "task_type": "chat",
            "managed_by_app": True,
        }
        with patch("app.main.list_running_servers", return_value=[chat_server]):
            with patch(
                "app.main.get_status",
                return_value={"running": True, "vllm_port": 8001, "model": "Qwen/Qwen2.5-7B-Instruct"},
            ):
                resp = self.client.post(
                    "/v1/embeddings",
                    json={"model": "jinaai/jina-embeddings-v3", "input": "hello"},
                )
        self.assertEqual(resp.status_code, 503)
        self.assertIn("No active vLLM server found", resp.json().get("detail", ""))


if __name__ == "__main__":
    unittest.main()
