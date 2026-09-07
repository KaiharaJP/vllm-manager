"""vLLM インスタンスの GPU 移設ロジックのユニットテスト。"""

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


class MoveInstanceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-move-test-")
        self._prev = os.environ.get(_TEST_DATA_ENV)
        os.environ[_TEST_DATA_ENV] = self._tmpdir
        self.sm = _reload_server_manager()
        self.instance_id = "chat-main"
        config = self.sm.load_config(self.instance_id)
        config.update(
            {
                "model_id": "org/model",
                "instance_id": self.instance_id,
                "instance_name": "Chat Main",
                "gpu_devices": "0,1",
                "tensor_parallel_size": 2,
                "vllm_port": 8011,
            }
        )
        self.sm.save_config(config, instance_id=self.instance_id)
        self.sm._upsert_instance_registry(
            {
                "instance_id": self.instance_id,
                "instance_name": "Chat Main",
                "model_id": "org/model",
                "auto_restore": True,
            }
        )
        self.inventory = {
            index: {"total_mb": 100_000.0, "used_mb": 0.0, "free_mb": 100_000.0}
            for index in range(4)
        }

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(_TEST_DATA_ENV, None)
        else:
            os.environ[_TEST_DATA_ENV] = self._prev
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_target_requires_exactly_tensor_parallel_gpu_count(self):
        normalized, indices, error = self.sm._validate_move_target(
            " 2 ", tensor_parallel_size=2, inventory=self.inventory
        )

        self.assertIsNone(normalized)
        self.assertEqual(indices, [])
        self.assertIn("ちょうど 2 台", error)

    def test_overlap_is_rejected_without_stopping_source(self):
        with patch.object(
            self.sm, "get_instance_status", return_value={"running": True}
        ), patch.object(
            self.sm, "_read_gpu_inventory", return_value=self.inventory
        ), patch.object(self.sm, "stop_instance") as mock_stop:
            result = self.sm.move_instance(self.instance_id, "1,2")

        self.assertFalse(result["success"])
        self.assertIn("同じ GPU", result["message"])
        mock_stop.assert_not_called()

    def test_actual_engine_gpu_is_used_when_config_exposes_all_gpus(self):
        config = self.sm.load_config(self.instance_id)
        config["gpu_devices"] = "all"
        config["tensor_parallel_size"] = 1
        self.sm.save_config(config, instance_id=self.instance_id)

        with patch.object(
            self.sm, "get_instance_status", return_value={"running": True, "pid": 1234}
        ), patch.object(
            self.sm, "_read_gpu_inventory", return_value=self.inventory
        ), patch.object(
            self.sm, "_gpu_indices_for_process_tree", return_value=[1]
        ), patch.object(self.sm, "stop_instance") as mock_stop:
            result = self.sm.move_instance(self.instance_id, "1")

        self.assertFalse(result["success"])
        self.assertEqual(result["source_gpu_devices"], "1")
        self.assertIn("同じ GPU", result["message"])
        mock_stop.assert_not_called()

    def test_gpu_usage_includes_engine_core_child_process(self):
        usage_by_pid = {
            100: {
                "total_vram_mb": 10.0,
                "gpu_uuids": ["GPU-A"],
                "vram_by_gpu_uuid_mb": {"GPU-A": 10.0},
            },
            101: {
                "total_vram_mb": 5000.0,
                "gpu_uuids": ["GPU-B"],
                "vram_by_gpu_uuid_mb": {"GPU-B": 5000.0},
            },
        }
        with patch.object(self.sm, "_process_tree_pids", return_value={100, 101}):
            usage = self.sm._gpu_usage_for_process_tree(100, usage_by_pid)

        self.assertEqual(usage["total_vram_mb"], 5010.0)
        self.assertEqual(usage["vram_by_gpu_uuid_mb"]["GPU-B"], 5000.0)
        self.assertEqual(set(usage["gpu_uuids"]), {"GPU-A", "GPU-B"})

    def test_preflight_failure_leaves_source_running(self):
        with patch.object(
            self.sm, "get_instance_status", return_value={"running": True}
        ), patch.object(
            self.sm, "_read_gpu_inventory", return_value=self.inventory
        ), patch.object(
            self.sm, "_preflight_move_config", return_value="GPU 2: VRAM不足"
        ), patch.object(self.sm, "stop_instance") as mock_stop:
            result = self.sm.move_instance(self.instance_id, "2,3")

        self.assertFalse(result["success"])
        self.assertIn("事前チェック", result["message"])
        self.assertIn("移設元は停止していません。", result["steps"])
        mock_stop.assert_not_called()

    def test_success_restarts_same_instance_on_target_gpu(self):
        with patch.object(
            self.sm, "get_instance_status", return_value={"running": True}
        ), patch.object(
            self.sm, "_read_gpu_inventory", return_value=self.inventory
        ), patch.object(
            self.sm, "_preflight_move_config", return_value=None
        ), patch.object(
            self.sm,
            "stop_instance",
            return_value={"success": True, "message": "stopped"},
        ) as mock_stop, patch.object(
            self.sm,
            "start_server",
            return_value={"success": True, "message": "started", "steps": ["ready"]},
        ) as mock_start:
            result = self.sm.move_instance(self.instance_id, " 2, 3 ")

        self.assertTrue(result["success"])
        self.assertEqual(result["source_gpu_devices"], "0,1")
        self.assertEqual(result["target_gpu_devices"], "2,3")
        mock_stop.assert_called_once_with(self.instance_id)
        kwargs = mock_start.call_args.kwargs
        self.assertEqual(kwargs["instance_id"], self.instance_id)
        self.assertEqual(kwargs["gpu_devices"], "2,3")
        self.assertEqual(kwargs["tensor_parallel_size"], 2)
        self.assertEqual(kwargs["vllm_port"], 8011)
        self.assertFalse(kwargs["download_model"])

    def test_target_start_failure_rolls_back_to_source_gpu(self):
        with patch.object(
            self.sm, "get_instance_status", return_value={"running": True}
        ), patch.object(
            self.sm, "_read_gpu_inventory", return_value=self.inventory
        ), patch.object(
            self.sm, "_preflight_move_config", return_value=None
        ), patch.object(
            self.sm,
            "stop_instance",
            return_value={"success": True, "message": "stopped"},
        ), patch.object(
            self.sm,
            "start_server",
            side_effect=[
                {"success": False, "message": "target failed", "steps": []},
                {"success": True, "message": "source restored", "steps": ["restored"]},
            ],
        ) as mock_start:
            result = self.sm.move_instance(self.instance_id, "2,3")

        self.assertFalse(result["success"])
        self.assertTrue(result["rollback_attempted"])
        self.assertTrue(result["rollback_success"])
        self.assertIn("元の GPU 0,1", result["message"])
        self.assertEqual(mock_start.call_count, 2)
        self.assertEqual(mock_start.call_args_list[0].kwargs["gpu_devices"], "2,3")
        self.assertEqual(mock_start.call_args_list[1].kwargs["gpu_devices"], "0,1")


class MoveInstanceApiTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-move-api-test-")
        self._prev = os.environ.get(_TEST_DATA_ENV)
        os.environ[_TEST_DATA_ENV] = self._tmpdir
        _reload_server_manager()

        import app.main as main_module

        importlib.reload(main_module)
        from app.auth import require_admin
        from fastapi.testclient import TestClient

        main_module.app.dependency_overrides[require_admin] = lambda: {
            "username": "admin",
            "role": "admin",
        }
        self.main = main_module
        self.client = TestClient(main_module.app)

    def tearDown(self):
        self.main.app.dependency_overrides.clear()
        if self._prev is None:
            os.environ.pop(_TEST_DATA_ENV, None)
        else:
            os.environ[_TEST_DATA_ENV] = self._prev
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_move_endpoint_passes_instance_and_gpu_target(self):
        move_result = {
            "success": True,
            "message": "moved",
            "steps": [],
            "instance_id": "chat-main",
            "source_gpu_devices": "0,1",
            "target_gpu_devices": "2,3",
            "rollback_attempted": False,
            "rollback_success": None,
        }
        with patch("app.main.move_instance", return_value=move_result) as mock_move:
            response = self.client.post(
                "/api/instances/chat-main/move",
                json={"gpu_devices": "2,3"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["target_gpu_devices"], "2,3")
        mock_move.assert_called_once_with("chat-main", "2,3")


if __name__ == "__main__":
    unittest.main()
