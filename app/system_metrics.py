"""Host system metrics for dashboard."""

from __future__ import annotations

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


def get_system_metrics() -> dict[str, Any]:
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
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
        "disk": {
            "usage_percent": disk.percent,
            "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
            "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
        },
        "gpus": gpus,
        "gpu_processes": gpu_processes,
    }
