import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import model_manager


class HfDownloadDirTests(unittest.TestCase):
    def test_download_dir_is_hub_under_hf_home(self):
        with mock.patch.dict(os.environ, {"HF_HOME": "/x/hf"}, clear=False):
            self.assertEqual(model_manager.hf_download_dir(), "/x/hf/hub")

    def test_cached_snapshot_checks_hub_then_legacy(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []

            def fake_snapshot(repo_id, cache_dir, **kw):
                calls.append(cache_dir)
                if cache_dir == root:
                    return str(Path(root) / "snap")
                raise FileNotFoundError

            with mock.patch.dict(os.environ, {"HF_HOME": root}, clear=False), \
                    mock.patch.object(model_manager, "snapshot_download", fake_snapshot):
                self.assertEqual(model_manager._cached_snapshot_path("a/b"), Path(root) / "snap")
            self.assertEqual(calls, [str(Path(root) / "hub"), root])


if __name__ == "__main__":
    unittest.main()
