"""埋め込みモデルカタログのユニットテスト。"""

import importlib
import os
import shutil
import tempfile
import unittest

_TEST_DATA_ENV = "VLLM_MANAGER_DATA_DIR"


def _reload_model_manager():
    import app.model_manager as mm

    importlib.reload(mm)
    return mm


class EmbeddingCatalogTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-mgr-embed-test-")
        self._prev = os.environ.get(_TEST_DATA_ENV)
        os.environ[_TEST_DATA_ENV] = self._tmpdir
        self.mm = _reload_model_manager()

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(_TEST_DATA_ENV, None)
        else:
            os.environ[_TEST_DATA_ENV] = self._prev
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_custom_embedding_model_metadata(self):
        saved = self.mm.save_model(
            {
                "id": "org/custom-embedding",
                "name": "Custom Embedding",
                "size": "1B",
                "task_type": "embedding",
                "recommended_context_length": 8192,
                "output_dimension": 1024,
                "license_note": "NonCommercial",
                "trust_remote_code": True,
            }
        )
        self.assertEqual(saved["task_type"], "embedding")
        self.assertEqual(saved["output_dimension"], 1024)
        self.assertEqual(saved["license_note"], "NonCommercial")

        catalog = self.mm.load_model_catalog()
        custom = next(item for item in catalog if item["id"] == "org/custom-embedding")
        self.assertEqual(custom["recommended_context_length"], 8192)
        self.assertEqual(custom["output_dimension"], 1024)
        self.assertEqual(custom["license_note"], "NonCommercial")

    def test_migrate_misclassified_rerank_models(self):
        self.mm.save_model(
            {
                "id": "BAAI/bge-reranker-v2-m3",
                "name": "BGE Reranker",
                "task_type": "embedding",
            }
        )
        self.mm.save_model(
            {
                "id": "cl-nagoya/ruri-v3-310m",
                "name": "Ruri embed",
                "task_type": "embedding",
            }
        )
        changed = self.mm.migrate_misclassified_rerank_models()
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["id"], "BAAI/bge-reranker-v2-m3")
        catalog = {item["id"]: item for item in self.mm.load_model_catalog()}
        self.assertEqual(catalog["BAAI/bge-reranker-v2-m3"]["task_type"], "rerank")
        self.assertEqual(catalog["cl-nagoya/ruri-v3-310m"]["task_type"], "embedding")


if __name__ == "__main__":
    unittest.main()
