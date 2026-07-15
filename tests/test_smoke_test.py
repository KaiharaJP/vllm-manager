"""起動後の自動疎通確認（スモークテスト）のユニットテスト。"""

from __future__ import annotations

import asyncio
import json
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

import app.server_manager as server_manager
from app.health_watchdog import HealthWatchdog


class _FakeChatHandler(BaseHTTPRequestHandler):
    response_body: dict = {
        "choices": [{"message": {"role": "assistant", "content": "2"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1, "prompt_tokens": 10, "total_tokens": 11},
    }

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps(self.response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence test output
        pass


class _FakeEmbeddingHandler(BaseHTTPRequestHandler):
    embedding_dim = 1792

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps(
            {
                "data": [{"embedding": [0.1] * self.embedding_dim, "index": 0}],
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _FakeRerankHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"results": [{"index": 0, "relevance_score": 0.91}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _FailingHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(500)
        self.end_headers()
        self.wfile.write(b"internal error")

    def log_message(self, *args):
        pass


def _run_server(handler_cls) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


class SmokeTestTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_smoke_test_success_for_chat_instance(self):
        httpd = _run_server(_FakeChatHandler)
        try:
            port = httpd.server_address[1]
            fake_status = {
                "running": True,
                "healthy": True,
                "task_type": "chat",
                "vllm_port": port,
                "model": "test-model",
            }
            with patch.object(server_manager, "get_instance_status", return_value=fake_status):
                result = self._run(server_manager.run_smoke_test("inst-1"))
        finally:
            httpd.shutdown()

        self.assertTrue(result["success"])
        self.assertEqual(result["task_type"], "chat")
        self.assertEqual(result["response_preview"], "2")
        self.assertIsNotNone(result["latency_ms"])
        self.assertEqual(result["tokens_generated"], 1)

    def test_smoke_test_success_for_embedding_instance(self):
        httpd = _run_server(_FakeEmbeddingHandler)
        try:
            port = httpd.server_address[1]
            fake_status = {
                "running": True,
                "healthy": True,
                "task_type": "embedding",
                "vllm_port": port,
                "model": "org/custom-embedding",
            }
            with patch.object(server_manager, "get_instance_status", return_value=fake_status):
                result = self._run(server_manager.run_smoke_test("embed-1"))
        finally:
            httpd.shutdown()

        self.assertTrue(result["success"])
        self.assertEqual(result["task_type"], "embedding")
        self.assertEqual(result["response_preview"], "embedding dim=1792")
        self.assertIsNotNone(result["latency_ms"])
        self.assertEqual(result["tokens_generated"], 8)

    def test_smoke_test_success_for_rerank_instance(self):
        httpd = _run_server(_FakeRerankHandler)
        try:
            port = httpd.server_address[1]
            fake_status = {
                "running": True,
                "healthy": True,
                "task_type": "rerank",
                "vllm_port": port,
                "model": "BAAI/bge-reranker-v2-m3",
            }
            with patch.object(server_manager, "get_instance_status", return_value=fake_status):
                result = self._run(server_manager.run_smoke_test("rerank-1"))
        finally:
            httpd.shutdown()

        self.assertTrue(result["success"])
        self.assertEqual(result["task_type"], "rerank")
        self.assertIn("rerank results=", result["response_preview"])

    def test_smoke_test_reports_upstream_failure(self):
        httpd = _run_server(_FailingHandler)
        try:
            port = httpd.server_address[1]
            fake_status = {
                "running": True,
                "healthy": True,
                "task_type": "chat",
                "vllm_port": port,
                "model": "test-model",
            }
            with patch.object(server_manager, "get_instance_status", return_value=fake_status):
                result = self._run(server_manager.run_smoke_test("inst-2"))
        finally:
            httpd.shutdown()

        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_smoke_test_skips_when_not_running(self):
        fake_status = {"running": False, "healthy": False}
        with patch.object(server_manager, "get_instance_status", return_value=fake_status):
            result = self._run(server_manager.run_smoke_test("inst-3"))
        self.assertFalse(result["success"])
        self.assertIn("起動していません", result["error"])

    def test_smoke_test_skips_when_unhealthy(self):
        fake_status = {"running": True, "healthy": False}
        with patch.object(server_manager, "get_instance_status", return_value=fake_status):
            result = self._run(server_manager.run_smoke_test("inst-4"))
        self.assertFalse(result["success"])
        self.assertIn("ヘルスチェック未通過", result["error"])


class HealthWatchdogSmokeTestTriggerTests(unittest.TestCase):
    def test_smoke_test_runs_once_when_instance_first_becomes_healthy(self):
        published: list[tuple[str, dict]] = []
        smoke_calls: list[str] = []

        async def fake_publisher(event_type, data, *, message, actor="system"):
            published.append((event_type, data))
            return {}

        async def fake_smoke_runner(instance_id: str) -> dict:
            smoke_calls.append(instance_id)
            return {"instance_id": instance_id, "success": True, "latency_ms": 42.0}

        watchdog = HealthWatchdog(event_publisher=fake_publisher, smoke_test_runner=fake_smoke_runner)

        with patch("app.server_manager.list_instances", return_value=[
            {"instance_id": "inst-a", "running": True, "healthy": True}
        ]):
            asyncio.run(watchdog._check_once())
            asyncio.run(watchdog._check_once())

        self.assertEqual(smoke_calls, ["inst-a"])
        smoke_events = [d for t, d in published if t == "instance_smoke_test"]
        self.assertEqual(len(smoke_events), 1)

    def test_smoke_test_state_cleared_when_instance_stops(self):
        async def fake_publisher(event_type, data, *, message, actor="system"):
            return {}

        async def fake_smoke_runner(instance_id: str) -> dict:
            return {"instance_id": instance_id, "success": True}

        watchdog = HealthWatchdog(event_publisher=fake_publisher, smoke_test_runner=fake_smoke_runner)

        with patch("app.server_manager.list_instances", return_value=[
            {"instance_id": "inst-b", "running": True, "healthy": True}
        ]):
            asyncio.run(watchdog._check_once())
        self.assertIn("inst-b", watchdog._smoke_tested)

        with patch("app.server_manager.list_instances", return_value=[]):
            asyncio.run(watchdog._check_once())
        self.assertNotIn("inst-b", watchdog._smoke_tested)


if __name__ == "__main__":
    unittest.main()
