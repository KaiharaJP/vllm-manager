"""ディスク使用率の重複排除ロジックのユニットテスト。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import app.system_metrics as system_metrics


class _FakeUsage:
    def __init__(self, total: int, used: int, percent: float):
        self.total = total
        self.used = used
        self.percent = percent


class _FakeStat:
    def __init__(self, st_dev: int):
        self.st_dev = st_dev


class ReadDisksDedupTests(unittest.TestCase):
    def test_same_underlying_device_is_deduplicated(self):
        """named volume が別ディレクトリでも、同一ディスク(st_dev)なら1件にまとめる。"""

        def fake_stat(path):
            # "/", HF_HOME, VLLM_MANAGER_DATA_DIR がすべて同じディスク上にある
            # （Docker named volume がホストの同一パーティションに作られるケース）。
            return _FakeStat(st_dev=100)

        with (
            patch.object(system_metrics.os, "stat", side_effect=fake_stat),
            patch.object(
                system_metrics.psutil,
                "disk_usage",
                return_value=_FakeUsage(total=1024**3 * 100, used=1024**3 * 45, percent=45.0),
            ),
            patch.dict(
                system_metrics.os.environ,
                {
                    "HF_HOME": "/app/hf-cache",
                    "VLLM_MANAGER_DATA_DIR": "/app/data",
                },
                clear=False,
            ),
        ):
            disks = system_metrics._read_disks()

        self.assertEqual(len(disks), 1)
        self.assertEqual(disks[0]["label"], "root")

    def test_identical_usage_bytes_deduped_even_with_different_st_dev(self):
        """コンテナの overlay2 ルートと同一ディスク上の named volume は
        st_dev が異なって見えても、使用量バイトが完全一致するなら重複排除する。
        """

        devices = {
            "/": 56,
            "/app/hf-cache": 66308,
            "/app/data": 66308,
        }

        def fake_stat(path):
            return _FakeStat(st_dev=devices.get(path, 99))

        with (
            patch.object(system_metrics.os, "stat", side_effect=fake_stat),
            patch.object(
                system_metrics.psutil,
                "disk_usage",
                return_value=_FakeUsage(total=1024**3 * 100, used=1024**3 * 45, percent=45.6),
            ),
            patch.dict(
                system_metrics.os.environ,
                {
                    "HF_HOME": "/app/hf-cache",
                    "VLLM_MANAGER_DATA_DIR": "/app/data",
                },
                clear=False,
            ),
        ):
            disks = system_metrics._read_disks()

        self.assertEqual(len(disks), 1)
        self.assertEqual(disks[0]["label"], "root")
        self.assertNotIn("_raw_total", disks[0])
        self.assertNotIn("_raw_used", disks[0])

    def test_different_devices_are_kept_separate(self):
        """本当に別ディスク（別 st_dev かつ使用量も異なる）の場合は個別に返す。"""

        devices = {
            "/": 1,
            "/app/hf-cache": 2,
            "/app/data": 3,
        }
        usages = {
            "/": _FakeUsage(total=1024**3 * 100, used=1024**3 * 10, percent=10.0),
            "/app/hf-cache": _FakeUsage(total=1024**3 * 500, used=1024**3 * 300, percent=60.0),
            "/app/data": _FakeUsage(total=1024**3 * 50, used=1024**3 * 5, percent=10.0),
        }

        def fake_stat(path):
            return _FakeStat(st_dev=devices.get(path, 99))

        def fake_disk_usage(path):
            return usages[path]

        with (
            patch.object(system_metrics.os, "stat", side_effect=fake_stat),
            patch.object(system_metrics.psutil, "disk_usage", side_effect=fake_disk_usage),
            patch.dict(
                system_metrics.os.environ,
                {
                    "HF_HOME": "/app/hf-cache",
                    "VLLM_MANAGER_DATA_DIR": "/app/data",
                },
                clear=False,
            ),
        ):
            disks = system_metrics._read_disks()

        self.assertEqual(len(disks), 3)
        self.assertEqual({d["label"] for d in disks}, {"root", "hf_cache", "vllm_data"})


if __name__ == "__main__":
    unittest.main()
