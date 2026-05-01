"""Model catalog and download job management."""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from app.event_bus import event_bus
from app.server_manager import DEFAULT_MODELS

DATA_DIR = Path(os.environ.get("VLLM_MANAGER_DATA_DIR", "/tmp/vllm-manager-data"))
MODELS_FILE = DATA_DIR / "models.json"
JOBS_FILE = DATA_DIR / "download_jobs.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    _ensure_data_dir()
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data: Any) -> None:
    _ensure_data_dir()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _cache_path_for(model_id: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    return hf_home / "hub" / f"models--{model_id.replace('/', '--')}"


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def load_model_catalog() -> list[dict[str, Any]]:
    saved = _read_json(MODELS_FILE, [])
    by_id = {
        item["id"]: {
            "id": item["id"],
            "name": item.get("name", item["id"]),
            "size": item.get("size", "unknown"),
            "revision": item.get("revision"),
            "gated": item.get("gated", False),
            "trust_remote_code": item.get("trust_remote_code", False),
            "recommended_context_length": item.get("recommended_context_length", 8192),
            "required_gpu_memory_gb": item.get("required_gpu_memory_gb"),
            "allowed_roles": item.get("allowed_roles", ["admin", "user"]),
            "source": item.get("source", "custom"),
        }
        for item in saved
    }
    for model in DEFAULT_MODELS:
        by_id.setdefault(
            model["id"],
            {
                **model,
                "revision": None,
                "gated": False,
                "trust_remote_code": False,
                "recommended_context_length": 8192,
                "required_gpu_memory_gb": None,
                "allowed_roles": ["admin", "user"],
                "source": "default",
            },
        )

    enriched = []
    for item in by_id.values():
        cache_path = _cache_path_for(item["id"])
        item["downloaded"] = cache_path.exists()
        item["cache_path"] = str(cache_path) if cache_path.exists() else None
        item["cache_size_bytes"] = _directory_size(cache_path)
        enriched.append(item)
    return sorted(enriched, key=lambda m: (m.get("source") != "default", m["name"]))


def save_model(model: dict[str, Any]) -> dict[str, Any]:
    catalog = {item["id"]: item for item in load_model_catalog()}
    catalog[model["id"]] = {
        "id": model["id"],
        "name": model.get("name") or model["id"],
        "size": model.get("size") or "unknown",
        "revision": model.get("revision") or None,
        "gated": bool(model.get("gated", False)),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
        "recommended_context_length": model.get("recommended_context_length") or 8192,
        "required_gpu_memory_gb": model.get("required_gpu_memory_gb"),
        "allowed_roles": model.get("allowed_roles") or ["admin", "user"],
        "source": model.get("source") or "custom",
    }
    custom_models = [item for item in catalog.values() if item.get("source") != "default"]
    _write_json(MODELS_FILE, custom_models)
    return catalog[model["id"]]


def load_jobs() -> list[dict[str, Any]]:
    return _read_json(JOBS_FILE, [])


def _save_job(job: dict[str, Any]) -> None:
    jobs = {item["id"]: item for item in load_jobs()}
    jobs[job["id"]] = job
    _write_json(JOBS_FILE, list(jobs.values()))


async def start_download_job(model_id: str, actor: str | None = None) -> dict[str, Any]:
    catalog = {item["id"]: item for item in load_model_catalog()}
    if model_id not in catalog:
        raise ValueError("model is not registered")

    job = {
        "id": str(uuid.uuid4()),
        "model_id": model_id,
        "status": "queued",
        "progress": 0,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "current_file": None,
        "message": "queued",
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "actor": actor,
    }
    _save_job(job)
    await event_bus.publish("model_download", job, message="queued", actor=actor)
    asyncio.create_task(_run_download_job(job, catalog[model_id]))
    return job


async def _run_download_job(job: dict[str, Any], model: dict[str, Any]) -> None:
    loop = asyncio.get_running_loop()

    async def update(**changes: Any) -> None:
        job.update(changes)
        job["updated_at"] = time.time()
        _save_job(job)
        await event_bus.publish("model_download", job, message=job.get("message"), actor=job.get("actor"))

    await update(status="running", message="download started")

    def emit_from_thread(**changes: Any) -> None:
        asyncio.run_coroutine_threadsafe(update(**changes), loop)

    class WebProgress(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            emit_from_thread(
                total_bytes=int(self.total or 0),
                downloaded_bytes=int(self.n or 0),
                current_file=str(self.desc or ""),
                message="downloading",
            )

        def update(self, n=1):
            result = super().update(n)
            total = int(self.total or job.get("total_bytes") or 0)
            done = int(self.n or 0)
            progress = int((done / total) * 100) if total else job.get("progress", 0)
            emit_from_thread(
                total_bytes=total,
                downloaded_bytes=done,
                progress=progress,
                current_file=str(self.desc or ""),
                message="downloading",
            )
            return result

    try:
        kwargs = {
            "repo_id": model["id"],
            "cache_dir": os.environ.get("HF_HOME", "/root/.cache/huggingface"),
            "revision": model.get("revision"),
            "token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
            "tqdm_class": WebProgress,
            "max_workers": int(os.environ.get("DOWNLOAD_WORKERS", "8")),
        }
        await asyncio.to_thread(snapshot_download, **{k: v for k, v in kwargs.items() if v is not None})
        await update(status="completed", progress=100, message="download completed")
    except Exception as exc:
        await update(status="failed", error=str(exc), message="download failed")
