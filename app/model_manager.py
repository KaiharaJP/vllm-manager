"""Model catalog and download job management."""

import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download
from tqdm.auto import tqdm

from app.event_bus import event_bus
from app.server_manager import DEFAULT_MODELS

DATA_DIR = Path(os.environ.get("VLLM_MANAGER_DATA_DIR", "/tmp/vllm-manager-data"))
MODELS_FILE = DATA_DIR / "models.json"
JOBS_FILE = DATA_DIR / "download_jobs.json"
REMOVED_DEFAULTS_FILE = DATA_DIR / "removed_default_models.json"


class DownloadCancelled(Exception):
    """Raised when a download job is cancelled by user action."""


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


def _load_removed_default_ids() -> set[str]:
    removed = _read_json(REMOVED_DEFAULTS_FILE, [])
    if not isinstance(removed, list):
        return set()
    return {item for item in removed if isinstance(item, str)}


def _save_removed_default_ids(removed_ids: set[str]) -> None:
    _write_json(REMOVED_DEFAULTS_FILE, sorted(removed_ids))


def _cache_path_candidates(model_id: str) -> list[Path]:
    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    model_dir = f"models--{model_id.replace('/', '--')}"
    # huggingface_hub versions differ:
    # - old: <HF_HOME>/hub/models--...
    # - new: <HF_HOME>/models--...
    return [hf_home / "hub" / model_dir, hf_home / model_dir]


def _cache_path_for(model_id: str) -> Path:
    for path in _cache_path_candidates(model_id):
        if path.exists():
            return path
    # keep old layout as display default when not downloaded yet
    return _cache_path_candidates(model_id)[0]


def _lock_path_candidates(model_id: str) -> list[Path]:
    hf_home = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    model_dir = f"models--{model_id.replace('/', '--')}"
    return [hf_home / ".locks" / model_dir, hf_home / "hub" / ".locks" / model_dir]


def _cached_snapshot_path(model_id: str, revision: str | None = None) -> Path | None:
    """Return snapshot path when the full model is locally cached."""
    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
            revision=revision,
            local_files_only=True,
        )
        return Path(snapshot_path)
    except Exception:
        return None


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _classify_download_error(exc: Exception) -> dict[str, str]:
    raw = str(exc)
    lowered = raw.lower()
    if "gated repo" in lowered or "access to model" in lowered and "restricted" in lowered:
        return {
            "error_code": "HF_GATED_REPO",
            "message": "download failed: gated model approval required",
            "error_hint": (
                "このモデルは Hugging Face でアクセス承認が必要です。"
                "モデルページでアクセス申請/承認後、承認済みトークンを HF_TOKEN に設定して再試行してください。"
            ),
        }
    if "401" in lowered and "huggingface" in lowered:
        return {
            "error_code": "HF_AUTH_REQUIRED",
            "message": "download failed: invalid or missing Hugging Face token",
            "error_hint": "HF_TOKEN が未設定または無効です。.env の HF_TOKEN を確認してください。",
        }
    if "repository not found" in lowered or "404" in lowered and "huggingface" in lowered:
        return {
            "error_code": "HF_REPO_NOT_FOUND",
            "message": "download failed: repository not found",
            "error_hint": "repo_id が正しいか確認してください（例: organization/model-name）。",
        }
    return {
        "error_code": "HF_DOWNLOAD_ERROR",
        "message": "download failed",
        "error_hint": "詳細は error フィールドを確認してください。",
    }


def _preferred_gguf_file(model_id: str, revision: str | None = None) -> str | None:
    """Pick a single GGUF file to avoid downloading every quantization variant."""
    try:
        files = list_repo_files(
            repo_id=model_id,
            revision=revision,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        )
    except Exception:
        return None

    gguf_files = sorted([path for path in files if path.lower().endswith(".gguf")])
    if not gguf_files:
        return None

    preferred_keywords = [
        "q4_k_m",
        "q4km",
        "q5_k_m",
        "q5km",
        "q8_0",
        "q8",
    ]
    lowered = [(path, path.lower()) for path in gguf_files]
    for keyword in preferred_keywords:
        for original, lower in lowered:
            if keyword in lower:
                return original
    return gguf_files[0]


def _current_incomplete_size_bytes(model_id: str) -> int:
    """Return the largest in-progress blob size for the model cache."""
    cache_dir = _cache_path_for(model_id) / "blobs"
    if not cache_dir.exists():
        return 0
    max_size = 0
    for blob in cache_dir.glob("*.incomplete"):
        try:
            max_size = max(max_size, blob.stat().st_size)
        except OSError:
            continue
    return max_size


def _repo_file_size_bytes(model_id: str, filename: str, revision: str | None = None) -> int:
    """Get remote file size from HF metadata when available."""
    try:
        info = HfApi().model_info(
            repo_id=model_id,
            revision=revision,
            files_metadata=True,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        )
    except Exception:
        return 0
    for sibling in info.siblings or []:
        if sibling.rfilename == filename and sibling.size:
            return int(sibling.size)
    return 0


def load_model_catalog() -> list[dict[str, Any]]:
    saved = _read_json(MODELS_FILE, [])
    removed_default_ids = _load_removed_default_ids()
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
        if model["id"] in removed_default_ids:
            continue
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
        snapshot_path = _cached_snapshot_path(item["id"], item.get("revision"))
        model_cache_dir = _cache_path_for(item["id"])
        item["downloaded"] = snapshot_path is not None
        item["cache_path"] = str(model_cache_dir) if snapshot_path else None
        item["cache_size_bytes"] = _directory_size(model_cache_dir) if snapshot_path else 0
        enriched.append(item)
    return sorted(enriched, key=lambda m: (m.get("source") != "default", m["name"]))


def save_model(model: dict[str, Any]) -> dict[str, Any]:
    removed_default_ids = _load_removed_default_ids()
    if model["id"] in removed_default_ids:
        removed_default_ids.discard(model["id"])
        _save_removed_default_ids(removed_default_ids)

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


def delete_model(model_id: str) -> dict[str, Any]:
    """Delete custom model registration and local cache."""
    saved = _read_json(MODELS_FILE, [])
    default_ids = {item["id"] for item in DEFAULT_MODELS}
    removed_default_ids = _load_removed_default_ids()
    before = len(saved)
    kept = [item for item in saved if item.get("id") != model_id]
    removed = before - len(kept)
    removed_default = False
    if removed:
        _write_json(MODELS_FILE, kept)
    elif model_id in default_ids and model_id not in removed_default_ids:
        removed_default_ids.add(model_id)
        _save_removed_default_ids(removed_default_ids)
        removed_default = True
    cache_result = delete_model_cache(model_id)
    return {
        "model_id": model_id,
        "removed": bool(removed or removed_default),
        "removed_entries": removed + (1 if removed_default else 0),
        "removed_default": removed_default,
        "cache_deleted": cache_result["deleted"],
        "cache_deleted_paths": cache_result["deleted_paths"],
        "bytes_freed": cache_result["bytes_freed"],
    }


def delete_model_cache(model_id: str) -> dict[str, Any]:
    """Delete local HF cache directories for a model."""
    deleted_paths: list[str] = []
    bytes_freed = 0
    candidates = _cache_path_candidates(model_id) + _lock_path_candidates(model_id)
    for path in candidates:
        if path.exists():
            bytes_freed += _directory_size(path)
            shutil.rmtree(path, ignore_errors=True)
            deleted_paths.append(str(path))
    return {
        "model_id": model_id,
        "deleted": bool(deleted_paths),
        "deleted_paths": deleted_paths,
        "bytes_freed": bytes_freed,
    }


def load_jobs() -> list[dict[str, Any]]:
    return _read_json(JOBS_FILE, [])


def _job_by_id(job_id: str) -> dict[str, Any] | None:
    for item in load_jobs():
        if item.get("id") == job_id:
            return item
    return None


def _active_jobs_for_model(model_id: str) -> list[dict[str, Any]]:
    return [
        item
        for item in load_jobs()
        if item.get("model_id") == model_id and item.get("status") in {"queued", "running"}
    ]


def _is_cancelled(job_id: str) -> bool:
    current = _job_by_id(job_id)
    return bool(current and current.get("status") == "cancelled")


def _save_job(job: dict[str, Any]) -> None:
    jobs = {item["id"]: item for item in load_jobs()}
    jobs[job["id"]] = job
    _write_json(JOBS_FILE, list(jobs.values()))


async def start_download_job(
    model_id: str,
    actor: str | None = None,
    force: bool = False,
    retry_count: int = 0,
    parent_job_id: str | None = None,
) -> dict[str, Any]:
    catalog = {item["id"]: item for item in load_model_catalog()}
    if model_id not in catalog:
        raise ValueError("model is not registered")
    active = _active_jobs_for_model(model_id)
    if active and not force:
        latest_active = sorted(active, key=lambda j: j.get("updated_at", 0), reverse=True)[0]
        raise ValueError(
            f"download already in progress for this model (job_id={latest_active.get('id')}, status={latest_active.get('status')})"
        )

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
        "error_code": None,
        "error_hint": None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "actor": actor,
        "retry_count": retry_count,
        "parent_job_id": parent_job_id,
    }
    _save_job(job)
    await event_bus.publish("model_download", job, message="queued", actor=actor)
    asyncio.create_task(_run_download_job(job, catalog[model_id]))
    return job


async def cancel_active_download_jobs(model_id: str, actor: str | None = None) -> dict[str, Any]:
    targets = _active_jobs_for_model(model_id)
    cancelled_ids: list[str] = []
    for job in targets:
        job["status"] = "cancelled"
        job["message"] = "download cancelled by user"
        job["updated_at"] = time.time()
        _save_job(job)
        cancelled_ids.append(job["id"])
        await event_bus.publish("model_download", job, message=job["message"], actor=actor)
    return {
        "model_id": model_id,
        "cancelled_count": len(cancelled_ids),
        "cancelled_job_ids": cancelled_ids,
    }


async def _run_download_job(job: dict[str, Any], model: dict[str, Any]) -> None:
    loop = asyncio.get_running_loop()

    async def update(**changes: Any) -> bool:
        current = _job_by_id(job["id"])
        if current:
            job.update(current)
        if job.get("status") == "cancelled" and changes.get("status") != "cancelled":
            return False
        job.update(changes)
        job["updated_at"] = time.time()
        _save_job(job)
        await event_bus.publish("model_download", job, message=job.get("message"), actor=job.get("actor"))
        return True

    await update(status="running", message="download started")

    def emit_from_thread(**changes: Any) -> None:
        asyncio.run_coroutine_threadsafe(update(**changes), loop)

    class WebProgress(tqdm):
        def __init__(self, *args, **kwargs):
            if _is_cancelled(job["id"]):
                raise DownloadCancelled("download cancelled")
            super().__init__(*args, **kwargs)
            emit_from_thread(
                total_bytes=int(self.total or 0),
                downloaded_bytes=int(self.n or 0),
                current_file=str(self.desc or ""),
                message="downloading",
            )

        def update(self, n=1):
            if _is_cancelled(job["id"]):
                raise DownloadCancelled("download cancelled")
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
        cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        kwargs = {
            "repo_id": model["id"],
            "cache_dir": cache_dir,
            "revision": model.get("revision"),
            "token": token,
            "tqdm_class": WebProgress,
            "max_workers": int(os.environ.get("DOWNLOAD_WORKERS", "8")),
        }
        # For GGUF repositories, download one preferred GGUF file instead of the full repository.
        if model["id"].lower().endswith("-gguf"):
            preferred = _preferred_gguf_file(model["id"], model.get("revision"))
            if preferred:
                await update(message=f"downloading selected GGUF file: {preferred}")
                total_bytes = _repo_file_size_bytes(model["id"], preferred, model.get("revision"))
                if total_bytes > 0:
                    await update(total_bytes=total_bytes)
                # Use single-file download for stability in proxy environments.
                worker_error: list[Exception] = []
                done = threading.Event()

                def run_download() -> None:
                    try:
                        hf_hub_download(
                            repo_id=model["id"],
                            filename=preferred,
                            revision=model.get("revision"),
                            cache_dir=cache_dir,
                            token=token,
                            force_download=False,
                        )
                    except Exception as exc:  # pragma: no cover - defensive capture
                        worker_error.append(exc)
                    finally:
                        done.set()

                threading.Thread(target=run_download, daemon=True).start()
                last_size = -1
                last_progress_at = time.time()
                stall_timeout_sec = int(os.environ.get("DOWNLOAD_STALL_TIMEOUT_SEC", "120"))
                max_auto_retry = int(os.environ.get("DOWNLOAD_AUTO_RETRY_MAX", "3"))
                while not done.is_set():
                    size = _current_incomplete_size_bytes(model["id"])
                    if size != last_size:
                        progress = int((size / total_bytes) * 100) if total_bytes else 0
                        await update(
                            downloaded_bytes=size,
                            progress=progress,
                            message=f"downloading selected GGUF file: {preferred}",
                        )
                        last_size = size
                        last_progress_at = time.time()
                    elif time.time() - last_progress_at >= stall_timeout_sec:
                        retry_count = int(job.get("retry_count", 0))
                        await update(
                            status="failed",
                            error_code="DOWNLOAD_STALLED",
                            error_hint=(
                                f"{stall_timeout_sec}秒以上進捗が止まったため、自動再開を試みます。"
                                if retry_count < max_auto_retry
                                else "進捗停止を検知しました。手動で再開してください。"
                            ),
                            message="download stalled",
                        )
                        if retry_count < max_auto_retry:
                            await start_download_job(
                                model["id"],
                                actor=job.get("actor"),
                                force=True,
                                retry_count=retry_count + 1,
                                parent_job_id=job["id"],
                            )
                        return
                    await asyncio.sleep(2)

                if worker_error:
                    raise worker_error[0]
                final_size = _current_incomplete_size_bytes(model["id"])
                await update(status="completed", progress=100, message="download completed")
                if final_size > 0:
                    await update(downloaded_bytes=final_size)
                return
        await asyncio.to_thread(snapshot_download, **{k: v for k, v in kwargs.items() if v is not None})
        await update(status="completed", progress=100, message="download completed")
    except DownloadCancelled:
        await update(status="cancelled", message="download cancelled by user")
    except Exception as exc:
        if _is_cancelled(job["id"]):
            await update(status="cancelled", message="download cancelled by user")
            return
        classified = _classify_download_error(exc)
        await update(
            status="failed",
            error=str(exc),
            error_code=classified["error_code"],
            error_hint=classified["error_hint"],
            message=classified["message"],
        )
