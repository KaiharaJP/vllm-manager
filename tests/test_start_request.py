"""HTTP 起動リクエスト正規化のユニットテスト。"""

import unittest

from app.main import ServerStartRequest, _resolve_start_request


class ResolveStartRequestTests(unittest.TestCase):
    def test_embedding_defaults_from_catalog(self):
        req = ServerStartRequest(model_id="jinaai/jina-embeddings-v3")
        resolved = _resolve_start_request(
            req,
            {
                "task_type": "embedding",
                "recommended_context_length": 8192,
                "trust_remote_code": False,
            },
        )
        self.assertEqual(resolved["task_type"], "embedding")
        self.assertEqual(resolved["context_length"], 8192)
        self.assertTrue(resolved["create_new_instance"])

    def test_explicit_task_type_overrides_catalog(self):
        req = ServerStartRequest(model_id="x", task_type="chat")
        resolved = _resolve_start_request(req, {"task_type": "embedding"})
        self.assertEqual(resolved["task_type"], "chat")
        self.assertFalse(resolved["create_new_instance"])

    def test_trust_remote_code_from_catalog_when_omitted(self):
        req = ServerStartRequest(model_id="x")
        resolved = _resolve_start_request(req, {"trust_remote_code": True})
        self.assertTrue(resolved["trust_remote_code"])

    def test_context_length_explicit_wins(self):
        req = ServerStartRequest(model_id="x", context_length=4096)
        resolved = _resolve_start_request(
            req,
            {"task_type": "embedding", "recommended_context_length": 8192},
        )
        self.assertEqual(resolved["context_length"], 4096)

    def test_rerank_defaults_from_catalog(self):
        req = ServerStartRequest(model_id="BAAI/bge-reranker-v2-m3")
        resolved = _resolve_start_request(
            req,
            {
                "task_type": "rerank",
                "recommended_context_length": 8192,
                "trust_remote_code": False,
            },
        )
        self.assertEqual(resolved["task_type"], "rerank")
        self.assertEqual(resolved["context_length"], 8192)
        self.assertTrue(resolved["create_new_instance"])


if __name__ == "__main__":
    unittest.main()
