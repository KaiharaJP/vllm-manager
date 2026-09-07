"""管理対象 vLLM インスタンス間の GPU スワップのユニットテスト。"""

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


class SwapInstanceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-swap-test-")
        self._prev = os.environ.get(_TEST_DATA_ENV)
        os.environ[_TEST_DATA_ENV] = self._tmpdir
        self.sm = _reload_server_manager()
        self.inventory = {
            index: {"total_mb": 100_000.0, "used_mb": 0.0, "free_mb": 100_000.0}
            for index in (0, 1)
        }
        for instance_id, gpu_devices, port in (
            ("instance-gpu0", "0", 8011),
            ("instance-gpu1", "1", 8012),
        ):
            config = self.sm.load_config(instance_id)
            config.update(
                {
                    "model_id": f"org/{instance_id}",
                    "instance_id": instance_id,
                    "instance_name": instance_id,
                    "gpu_devices": gpu_devices,
                    "tensor_parallel_size": 1,
                    "vllm_port": port,
                }
            )
            self.sm.save_config(config, instance_id=instance_id)
            self.sm._upsert_instance_registry(
                {
                    "instance_id": instance_id,
                    "instance_name": instance_id,
                    "model_id": config["model_id"],
                    "auto_restore": True,
                }
            )

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(_TEST_DATA_ENV, None)
        else:
            os.environ[_TEST_DATA_ENV] = self._prev
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_swap_restarts_both_instances_on_each_others_gpu(self):
        statuses = [
            {"running": True, "pid": 100},
            {"running": True, "pid": 101},
        ]
        with patch.object(
            self.sm, "get_instance_status", side_effect=statuses
        ), patch.object(
            self.sm, "_read_gpu_inventory", return_value=self.inventory
        ), patch.object(
            self.sm, "_gpu_indices_for_process_tree", side_effect=([0], [1])
        ), patch.object(
            self.sm, "_preflight_move_config", return_value=None
        ), patch.object(
            self.sm,
            "stop_instance",
            return_value={"success": True, "message": "stopped"},
        ) as mock_stop, patch.object(
            self.sm,
            "start_server",
            return_value={"success": True, "message": "started", "steps": []},
        ) as mock_start:
            result = self.sm.swap_instances("instance-gpu0", "instance-gpu1")

        self.assertTrue(result["success"])
        self.assertEqual(result["first_source_gpu_devices"], "0")
        self.assertEqual(result["second_source_gpu_devices"], "1")
        self.assertEqual(mock_stop.call_args_list[0].args, ("instance-gpu0",))
        self.assertEqual(mock_stop.call_args_list[1].args, ("instance-gpu1",))
        self.assertEqual(mock_start.call_count, 2)
        self.assertEqual(mock_start.call_args_list[0].kwargs["instance_id"], "instance-gpu0")
        self.assertEqual(mock_start.call_args_list[0].kwargs["gpu_devices"], "1")
        self.assertEqual(mock_start.call_args_list[1].kwargs["instance_id"], "instance-gpu1")
        self.assertEqual(mock_start.call_args_list[1].kwargs["gpu_devices"], "0")

    def test_swap_rejects_instances_sharing_a_gpu(self):
        statuses = [
            {"running": True, "pid": 100},
            {"running": True, "pid": 101},
        ]
        with patch.object(
            self.sm, "get_instance_status", side_effect=statuses
        ), patch.object(
            self.sm, "_read_gpu_inventory", return_value=self.inventory
        ), patch.object(
            self.sm, "_gpu_indices_for_process_tree", side_effect=([0], [0])
        ), patch.object(self.sm, "stop_instance") as mock_stop:
            result = self.sm.swap_instances("instance-gpu0", "instance-gpu1")

        self.assertFalse(result["success"])
        self.assertIn("同じGPU", result["message"])
        mock_stop.assert_not_called()


class SwapInstanceApiTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-swap-api-test-")
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

    def test_swap_endpoint_passes_both_instance_ids(self):
        result = {
            "success": True,
            "message": "swapped",
            "steps": [],
            "first_instance_id": "instance-gpu0",
            "second_instance_id": "instance-gpu1",
            "first_source_gpu_devices": "0",
            "second_source_gpu_devices": "1",
            "rollback_attempted": False,
            "rollback_success": None,
        }
        with patch("app.main.swap_instances", return_value=result) as mock_swap:
            response = self.client.post(
                "/api/instances/swap",
                json={
                    "first_instance_id": "instance-gpu0",
                    "second_instance_id": "instance-gpu1",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["message"], "swapped")
        mock_swap.assert_called_once_with("instance-gpu0", "instance-gpu1")


if __name__ == "__main__":
    unittest.main()
