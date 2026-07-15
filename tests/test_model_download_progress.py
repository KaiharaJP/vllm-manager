"""Download progress helper tests."""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import app.model_manager as model_manager


class ModelDownloadProgressTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="vllm-download-test-")
        self._prev_data_dir = model_manager.DATA_DIR
        self._prev_jobs_file = model_manager.JOBS_FILE
        self._prev_models_file = model_manager.MODELS_FILE
        model_manager.DATA_DIR = Path(self._tmpdir)
        model_manager.JOBS_FILE = model_manager.DATA_DIR / "download_jobs.json"
        model_manager.MODELS_FILE = model_manager.DATA_DIR / "models.json"
        model_manager._write_json(
            model_manager.MODELS_FILE,
            [
                {
                    "id": "org/demo-model",
                    "name": "demo",
                    "size": "1GB",
                    "task_type": "chat",
                    "source": "custom",
                }
            ],
        )

    def tearDown(self):
        model_manager.DATA_DIR = self._prev_data_dir
        model_manager.JOBS_FILE = self._prev_jobs_file
        model_manager.MODELS_FILE = self._prev_models_file
        model_manager._last_disk_bytes.clear()
        from app import download_worker

        download_worker._active_processes.clear()

    def test_cancel_active_download_jobs_terminates_worker(self):
        running_job = {
            "id": "job-run",
            "model_id": "org/demo-model",
            "status": "running",
            "updated_at": time.time(),
        }
        model_manager._write_json(model_manager.JOBS_FILE, [running_job])

        async def run():
            with mock.patch.object(model_manager, "_terminate_download_worker", mock.AsyncMock()) as terminate_mock:
                with mock.patch.object(model_manager.event_bus, "publish", mock.AsyncMock()):
                    result = await model_manager.cancel_active_download_jobs("org/demo-model", actor="test")
            return result, terminate_mock

        result, terminate_mock = asyncio.run(run())
        self.assertEqual(result["cancelled_count"], 1)
        terminate_mock.assert_awaited_once_with("org/demo-model")

    def test_resume_download_job_terminates_worker_before_restart(self):
        stale_job = {
            "id": "job-old",
            "model_id": "org/demo-model",
            "status": "running",
            "retry_count": 1,
            "updated_at": time.time(),
        }
        model_manager._write_json(model_manager.JOBS_FILE, [stale_job])

        async def run():
            with mock.patch.object(model_manager, "_terminate_download_worker", mock.AsyncMock()) as terminate_mock:
                with mock.patch.object(model_manager, "cancel_active_download_jobs", mock.AsyncMock(return_value={"cancelled_count": 1})):
                    with mock.patch.object(model_manager, "start_download_job", mock.AsyncMock(return_value={"id": "job-new"})) as start_mock:
                        job = await model_manager.resume_download_job("org/demo-model", actor="user")
            return terminate_mock, start_mock, job

        terminate_mock, start_mock, job = asyncio.run(run())
        terminate_mock.assert_awaited_once_with("org/demo-model")
        start_mock.assert_awaited_once()
        self.assertEqual(job["id"], "job-new")
        self.assertEqual(start_mock.await_args.kwargs["retry_count"], 1)

    def test_spawn_download_worker_replaces_existing_process(self):
        from app import download_worker

        first = mock.Mock()
        first.is_alive.return_value = True
        first.terminate = mock.Mock()
        first.join = mock.Mock()
        first.kill = mock.Mock()
        second = mock.Mock()
        second.start = mock.Mock()

        with mock.patch.object(download_worker.mp, "get_context") as get_context:
            ctx = mock.Mock()
            ctx.Process.return_value = second
            get_context.return_value = ctx
            download_worker._active_processes["org/demo-model"] = first
            download_worker.spawn("org/demo-model", {"repo_id": "org/demo-model", "kind": "snapshot"})
        first.terminate.assert_called_once()
        second.start.assert_called_once()
        self.assertIs(download_worker._active_processes["org/demo-model"], second)

    def test_inspect_stalled_auto_resume_uses_single_restart_path(self):
        stale_job = {
            "id": "job-old",
            "model_id": "org/demo-model",
            "status": "running",
            "progress": 10,
            "downloaded_bytes": 100,
            "total_bytes": 1000,
            "updated_at": time.time() - 300,
            "retry_count": 0,
        }
        model_manager._write_json(model_manager.JOBS_FILE, [stale_job])
        model_manager._last_disk_bytes["org/demo-model"] = (100, time.time() - 300)

        async def run():
            with mock.patch.object(model_manager, "_download_progress_bytes", return_value=100):
                with mock.patch.object(model_manager, "_terminate_download_worker", mock.AsyncMock()) as terminate_mock:
                    with mock.patch.object(
                        model_manager,
                        "start_download_job",
                        mock.AsyncMock(return_value={"id": "job-new", "model_id": "org/demo-model", "status": "queued"}),
                    ) as start_mock:
                        with mock.patch.object(model_manager.event_bus, "publish", mock.AsyncMock()):
                            actions = await model_manager.inspect_stalled_download_jobs(actor="test")
            return actions, terminate_mock, start_mock

        actions, terminate_mock, start_mock = asyncio.run(run())
        self.assertEqual(actions[0]["action"], "auto_resumed")
        start_mock.assert_awaited_once()
        terminate_mock.assert_not_awaited()

    def test_tqdm_total_looks_like_bytes(self):
        self.assertFalse(model_manager._tqdm_total_looks_like_bytes(14))
        self.assertTrue(model_manager._tqdm_total_looks_like_bytes(4_904_765_056))

    def test_download_progress_bytes_uses_cache_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            model_id = "org/demo-model"
            model_dir = cache_root / "hub" / "models--org--demo-model"
            blobs = model_dir / "blobs"
            blobs.mkdir(parents=True)
            incomplete = blobs / "abc.incomplete"
            incomplete.write_bytes(b"x" * 2048)
            (model_dir / "refs").mkdir()
            (model_dir / "refs" / "main").write_text("snapshot")

            with mock.patch.object(model_manager, "_cache_path_candidates", return_value=[model_dir]):
                with mock.patch.dict("os.environ", {"HF_HOME": str(cache_root)}, clear=False):
                    self.assertGreaterEqual(model_manager._download_progress_bytes(model_id), 2048)

    def test_repo_total_size_bytes_sums_siblings(self):
        sibling = mock.Mock(size=1000)
        sibling.rfilename = "model.safetensors"
        info = mock.Mock(siblings=[sibling, mock.Mock(size=500, rfilename="config.json")])
        with mock.patch.object(model_manager.HfApi, "model_info", return_value=info):
            self.assertEqual(model_manager._repo_total_size_bytes("org/demo-model"), 1500)

    def test_apply_cache_progress_to_job_uses_disk_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            model_id = "org/demo-model"
            model_dir = cache_root / "hub" / "models--org--demo-model"
            blobs = model_dir / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "abc.incomplete").write_bytes(b"x" * 4096)

            with mock.patch.object(model_manager, "_cache_path_candidates", return_value=[model_dir]):
                with mock.patch.dict("os.environ", {"HF_HOME": str(cache_root)}, clear=False):
                    with mock.patch.object(model_manager, "_repo_total_size_bytes", return_value=8192):
                        job = {
                            "id": "job-1",
                            "model_id": model_id,
                            "status": "queued",
                            "progress": 0,
                            "downloaded_bytes": 0,
                            "total_bytes": 0,
                        }
                        model_manager._apply_cache_progress_to_job(
                            job,
                            {"id": model_id, "revision": None},
                        )
            self.assertEqual(job["downloaded_bytes"], 4096)
            self.assertEqual(job["total_bytes"], 8192)
            self.assertEqual(job["progress"], 50)

    def test_update_never_regresses_downloaded_bytes(self):
        job = {
            "id": "job-mono",
            "model_id": "org/demo-model",
            "status": "running",
            "progress": 30,
            "downloaded_bytes": 1500,
            "total_bytes": 5000,
            "updated_at": time.time(),
        }
        model_manager._write_json(model_manager.JOBS_FILE, [job])

        async def run():
            with mock.patch.object(model_manager.event_bus, "publish", mock.AsyncMock()):
                saved_job = dict(job)

                async def update(**changes):
                    current = model_manager._job_by_id(saved_job["id"])
                    if current:
                        saved_job.update(current)
                    if "downloaded_bytes" in changes:
                        changes["downloaded_bytes"] = max(
                            int(changes["downloaded_bytes"] or 0),
                            int(saved_job.get("downloaded_bytes") or 0),
                        )
                    if "progress" in changes:
                        changes["progress"] = max(
                            int(changes["progress"] or 0),
                            int(saved_job.get("progress") or 0),
                        )
                    saved_job.update(changes)
                    model_manager._save_job(saved_job)
                    return True

                await update(downloaded_bytes=0, progress=1)
            return model_manager.load_jobs()[0]

        saved = asyncio.run(run())
        self.assertEqual(saved["downloaded_bytes"], 1500)
        self.assertEqual(saved["progress"], 30)

    def test_inspect_stalled_download_jobs_auto_resumes(self):
        stale_job = {
            "id": "job-old",
            "model_id": "org/demo-model",
            "status": "running",
            "progress": 10,
            "downloaded_bytes": 100,
            "total_bytes": 1000,
            "updated_at": time.time() - 300,
            "retry_count": 0,
        }
        model_manager._write_json(model_manager.JOBS_FILE, [stale_job])
        model_manager._last_disk_bytes["org/demo-model"] = (100, time.time() - 300)

        async def run():
            with mock.patch.object(model_manager, "_download_progress_bytes", return_value=100):
                with mock.patch.object(model_manager, "start_download_job", mock.AsyncMock(return_value={"id": "job-new", "model_id": "org/demo-model", "status": "queued"})) as start_mock:
                    with mock.patch.object(model_manager.event_bus, "publish", mock.AsyncMock()):
                        actions = await model_manager.inspect_stalled_download_jobs(actor="test")
            return actions, start_mock

        actions, start_mock = asyncio.run(run())
        self.assertEqual(actions[0]["action"], "auto_resumed")
        start_mock.assert_awaited_once()
        self.assertEqual(start_mock.await_args.kwargs["retry_count"], 1)

    def test_inspect_stalled_download_jobs_syncs_disk_progress(self):
        stale_job = {
            "id": "job-sync",
            "model_id": "org/demo-model",
            "status": "running",
            "progress": 10,
            "downloaded_bytes": 100,
            "total_bytes": 1000,
            "updated_at": time.time() - 300,
            "retry_count": 0,
        }
        model_manager._write_json(model_manager.JOBS_FILE, [stale_job])

        async def run():
            with mock.patch.object(model_manager, "_download_progress_bytes", return_value=250):
                with mock.patch.object(model_manager.event_bus, "publish", mock.AsyncMock()):
                    return await model_manager.inspect_stalled_download_jobs(actor="test")

        actions = asyncio.run(run())
        self.assertEqual(actions[0]["action"], "synced_progress")
        saved = model_manager.load_jobs()[0]
        self.assertEqual(saved["downloaded_bytes"], 250)
        self.assertEqual(saved["progress"], 25)

    def test_clamp_progress_caps_at_100(self):
        self.assertEqual(model_manager._clamp_progress(199), 100)
        self.assertEqual(model_manager._clamp_progress(-1), 0)
        self.assertEqual(model_manager._clamp_progress(50), 50)

    def test_reconcile_orphan_marks_completed_when_snapshot_exists(self):
        job = {
            "id": "job-orphan",
            "model_id": "org/demo-model",
            "status": "running",
            "progress": 199,
            "downloaded_bytes": 9999,
            "total_bytes": 5000,
            "updated_at": time.time() - 600,
        }
        model_manager._write_json(model_manager.JOBS_FILE, [job])
        with mock.patch.object(model_manager, "_cached_snapshot_path", return_value=Path("/tmp/snap")):
            with mock.patch.object(model_manager, "_download_progress_bytes", return_value=5000):
                with mock.patch.object(model_manager, "_repo_total_size_bytes", return_value=5000):
                    actions = model_manager.reconcile_orphan_download_jobs(actor="test")
        self.assertEqual(actions[0]["action"], "marked_completed")
        saved = model_manager.load_jobs()[0]
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["progress"], 100)


if __name__ == "__main__":
    unittest.main()
