"""Host system metrics for dashboard."""

from __future__ import annotations

import os
import subprocess
from typing import Any

import psutil


def _gpu_uuid_to_index() -> dict[str, int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
    except Exception:
        return {}
    mapping: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mapping[parts[1]] = int(parts[0])
        except ValueError:
            continue
    return mapping


def _read_gpu_processes() -> list[dict[str, Any]]:
    uuid_to_index = _gpu_uuid_to_index()
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
    except Exception:
        return []
    processes: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            process_name = parts[1]
            used_memory_mb = float(parts[2])
            gpu_uuid = parts[3]
        except ValueError:
            continue
        cmdline = ""
        try:
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline())
        except Exception:
            pass
        processes.append(
            {
                "pid": pid,
                "process_name": process_name,
                "used_memory_mb": used_memory_mb,
                "gpu_uuid": gpu_uuid,
                "gpu_index": uuid_to_index.get(gpu_uuid),
                "cmdline": cmdline,
            }
        )
    processes.sort(key=lambda item: (item.get("gpu_index", -1), -float(item["used_memory_mb"])))
    return processes


def _read_gpu_metrics() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_percent": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                    "temperature_c": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return gpus


def _disk_usage(label: str, path: str) -> dict[str, Any] | None:
    try:
        usage = psutil.disk_usage(path)
    except OSError:
        return None
    return {
        "label": label,
        "path": path,
        "usage_percent": usage.percent,
        "used_gb": round(usage.used / 1024 / 1024 / 1024, 2),
        "total_gb": round(usage.total / 1024 / 1024 / 1024, 2),
        # 重複排除専用の生バイト値（レスポンスには含めない）
        "_raw_total": usage.total,
        "_raw_used": usage.used,
    }


def _mount_device_id(path: str) -> Any:
    """パスが乗っている実ファイルシステムを識別する ID（st_dev）を返す。

    ディレクトリのパス文字列が違っても、Docker の named volume が
    ホスト側で同一ディスク（同一パーティション）上に作られている場合は
    st_dev が一致するため、見た目上の別ボリュームでも正しく重複排除できる。
    `os.path.realpath` によるパス文字列比較だけでは、この「別ディレクトリだが
    同じディスク」のケースを検出できない。
    """
    try:
        return os.stat(path).st_dev
    except OSError:
        return None


def _read_disks() -> list[dict[str, Any]]:
    """ルートに加え、モデルキャッシュ・管理データの実マウント先の使用率も個別に返す。

    HF キャッシュ（大容量モデル格納）と vllm-data（設定/監査ログ/APIキー）は
    別ボリュームのことが多く、ルート("/")の使用率だけでは容量逼迫を見逃す。
    ただし named volume が実際にはホストの同一ディスク上にある場合は
    数値が重複するだけなので、次の2段階で重複排除する。

    1. st_dev（マウントの実体）が一致する場合 -> 同一とみなす
       （named volume 同士が同じホストパスにバインドされているケース）
    2. st_dev が一致しなくても、使用量（バイト単位、丸め前）が完全一致する場合
       -> 同一とみなす（コンテナ自身の overlay2 ルートと、ホスト側で
       同じディスクにバインドされた named volume は Linux 上は別デバイス
       として見えるが、実際には物理的に同じディスクであるケース）
    """
    seen_devices: set[Any] = set()
    seen_usages: set[tuple[int, int]] = set()
    disks: list[dict[str, Any]] = []

    def _try_add(label: str, path: str) -> None:
        device = _mount_device_id(path)
        if device is not None and device in seen_devices:
            return
        entry = _disk_usage(label, path)
        if not entry:
            return
        usage_key = (entry.pop("_raw_total"), entry.pop("_raw_used"))
        if usage_key in seen_usages:
            return
        disks.append(entry)
        seen_usages.add(usage_key)
        if device is not None:
            seen_devices.add(device)

    _try_add("root", "/")

    hf_home = os.environ.get("HF_HOME", "/app/hf-cache")
    data_dir = os.environ.get("VLLM_MANAGER_DATA_DIR", "/tmp/vllm-manager-data")

    for label, path in (("hf_cache", hf_home), ("vllm_data", data_dir)):
        _try_add(label, path)

    return disks


def get_system_metrics() -> dict[str, Any]:
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disks = _read_disks()
    disk = disks[0] if disks else {"usage_percent": 0.0, "used_gb": 0.0, "total_gb": 0.0}
    gpus = _read_gpu_metrics()
    gpu_processes = _read_gpu_processes()

    return {
        "cpu": {
            "usage_percent": cpu_percent,
            "cores_logical": psutil.cpu_count(logical=True),
            "cores_physical": psutil.cpu_count(logical=False),
        },
        "memory": {
            "usage_percent": memory.percent,
            "used_gb": round(memory.used / 1024 / 1024 / 1024, 2),
            "total_gb": round(memory.total / 1024 / 1024 / 1024, 2),
        },
        # 後方互換: 従来どおりルート("/")の使用率を返す
        "disk": {
            "usage_percent": disk["usage_percent"],
            "used_gb": disk["used_gb"],
            "total_gb": disk["total_gb"],
        },
        # 新規: root / hf_cache / vllm_data を個別に返す（実体が同じマウントなら重複しない）
        "disks": disks,
        "gpus": gpus,
        "gpu_processes": gpu_processes,
    }
