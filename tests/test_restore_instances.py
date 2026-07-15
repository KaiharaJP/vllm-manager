"""vLLM インスタンス自動復旧フラグのユニットテスト。"""

import importlib
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

_TEST_DATA_ENV = "VLLM_MANAGER_DATA_DIR"


def _reload_server_manager():
    import app.server_manager as sm

    importlib.reload(sm)
    return sm


class RestoreInstanceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-restore-test-")
        self._prev = os.environ.get(_TEST_DATA_ENV)
        os.environ[_TEST_DATA_ENV] = self._tmpdir
        self.sm = _reload_server_manager()

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(_TEST_DATA_ENV, None)
        else:
            os.environ[_TEST_DATA_ENV] = self._prev
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_stop_clears_auto_restore(self):
        instance_id = "chat-main"
        self.sm._upsert_instance_registry(
            {
                "instance_id": instance_id,
                "model_id": "org/model",
                "auto_restore": True,
            }
        )
        paths = self.sm._instance_paths(instance_id)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        paths["pid"].write_text("999999")

        with patch.object(self.sm, "_process_exists", return_value=False):
            result = self.sm.stop_instance(instance_id)

        self.assertTrue(result["success"])
        entry = self.sm._registry_entry_for(instance_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertFalse(entry.get("auto_restore"))

    def test_restore_skips_when_auto_restore_false(self):
        self.sm._upsert_instance_registry(
            {"instance_id": "stopped-one", "model_id": "org/model", "auto_restore": False}
        )
        with patch.object(self.sm, "start_server") as mock_start:
            results = self.sm.restore_managed_instances()
        mock_start.assert_not_called()
        self.assertEqual(results, [])

    def test_restore_starts_marked_instances(self):
        instance_id = "embed-a"
        self.sm._upsert_instance_registry(
            {"instance_id": instance_id, "model_id": "jina/embed", "auto_restore": True}
        )
        cfg = self.sm.load_config(instance_id)
        cfg["model_id"] = "jina/embed"
        cfg["task_type"] = "embedding"
        self.sm.save_config(cfg, instance_id=instance_id)

        with patch.object(self.sm, "get_instance_status", return_value={"running": False}), patch.object(
            self.sm, "start_server", return_value={"success": True, "message": "ok"}
        ) as mock_start:
            results = self.sm.restore_managed_instances()

        mock_start.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])


if __name__ == "__main__":
    unittest.main()
