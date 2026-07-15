"""Model catalog and download job management."""

import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, list_repo_files, snapshot_download

from app import download_worker
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
    hf_home = Path(os.environ.get("HF_HOME", "/app/hf-cache"))
    model_dir = f"models--{model_id.replace('/', '--')}"
    # huggingface_hub versions differ:
    # - old: <HF_HOME>/hub/models--...
    # - new: <HF_HOME>/models--...
    return [hf_home / "hub" / model_dir, hf_home / model_dir]


def _safe_exists(path: Path) -> bool:
    """`Path.exists()` だが、権限エラー等で API 全体を落とさないようにする。

    ホスト側の権限設定ミスやボリュームの所有者不整合があっても、
    モデル一覧取得のような読み取り専用エンドポイントは 500 にせず、
    「未ダウンロード扱い」として動作を継続する。
    """
    try:
        return path.exists()
    except PermissionError:
        return False
    except OSError:
        return False


def _cache_path_for(model_id: str) -> Path:
    for path in _cache_path_candidates(model_id):
        if _safe_exists(path):
            return path
    # keep old layout as display default when not downloaded yet
    return _cache_path_candidates(model_id)[0]


def _lock_path_candidates(model_id: str) -> list[Path]:
    hf_home = Path(os.environ.get("HF_HOME", "/app/hf-cache"))
    model_dir = f"models--{model_id.replace('/', '--')}"
    return [hf_home / ".locks" / model_dir, hf_home / "hub" / ".locks" / model_dir]


def _cached_snapshot_path(model_id: str, revision: str | None = None) -> Path | None:
    """Return snapshot path when the full model is locally cached."""
    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=os.environ.get("HF_HOME", "/app/hf-cache"),
            revision=revision,
            local_files_only=True,
        )
        return Path(snapshot_path)
    except Exception:
        return None


def _directory_size(path: Path) -> int:
    if not _safe_exists(path):
        return 0
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


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
    if "permission denied" in lowered or isinstance(exc, PermissionError):
        return {
            "error_code": "HF_CACHE_PERMISSION",
            "message": "download failed: huggingface cache permission denied",
            "error_hint": (
                "HF キャッシュ（.locks 等）への書き込み権限がありません。"
                "backend を再ビルド・再起動して docker-entrypoint の権限修復を適用するか、"
                "管理者がコンテナ内で .locks の所有権を vllmapp に揃えてください。"
            ),
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
    if not _safe_exists(cache_dir):
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


def _repo_total_size_bytes(model_id: str, revision: str | None = None) -> int:
    """Sum remote file sizes for byte-based download progress."""
    try:
        info = HfApi().model_info(
            repo_id=model_id,
            revision=revision,
            files_metadata=True,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        )
    except Exception:
        return 0
    total = 0
    for sibling in info.siblings or []:
        if sibling.size:
            total += int(sibling.size)
    return total


def _tqdm_total_looks_like_bytes(total: int | float | None) -> bool:
    """snapshot_download の tqdm はファイル数(例: 14)とバイト数の両方を返す。"""
    return bool(total) and int(total) >= 1_000_000


def _download_progress_bytes(model_id: str) -> int:
    """Return best-effort downloaded bytes while a snapshot is in progress."""
    cache_path = _cache_path_for(model_id)
    cache_size = _directory_size(cache_path) if _safe_exists(cache_path) else 0
    return max(cache_size, _current_incomplete_size_bytes(model_id))


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
            "output_dimension": item.get("output_dimension"),
            "license_note": item.get("license_note"),
            "allowed_roles": item.get("allowed_roles", ["admin", "user"]),
            "source": item.get("source", "custom"),
            "task_type": item.get("task_type", "chat"),
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
                "revision": model.get("revision"),
                "gated": model.get("gated", False),
                "trust_remote_code": model.get("trust_remote_code", False),
                "recommended_context_length": model.get("recommended_context_length", 8192),
                "required_gpu_memory_gb": model.get("required_gpu_memory_gb"),
                "output_dimension": model.get("output_dimension"),
                "license_note": model.get("license_note"),
                "allowed_roles": ["admin", "user"],
                "source": "default",
                "task_type": model.get("task_type", "chat"),
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
        "output_dimension": model.get("output_dimension"),
        "license_note": model.get("license_note"),
        "allowed_roles": model.get("allowed_roles") or ["admin", "user"],
        "source": model.get("source") or "custom",
        "task_type": model.get("task_type") or "chat",
    }
    custom_models = [item for item in catalog.values() if item.get("source") != "default"]
    _write_json(MODELS_FILE, custom_models)
    return catalog[model["id"]]


def _looks_like_reranker_model_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return "rerank" in lowered


def migrate_misclassified_rerank_models() -> list[dict[str, str]]:
    """embedding として登録された reranker を task_type=rerank に直す。

    名前に `rerank` を含むものだけ対象（ruri-v3-310m など embedding 系は触らない）。
    """
    saved = _read_json(MODELS_FILE, [])
    if not isinstance(saved, list):
        return []
    changed: list[dict[str, str]] = []
    updated: list[dict[str, Any]] = []
    for item in saved:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "")
        task_type = str(item.get("task_type") or "chat")
        if model_id and _looks_like_reranker_model_id(model_id) and task_type == "embedding":
            item = {**item, "task_type": "rerank"}
            changed.append({"id": model_id, "from": "embedding", "to": "rerank"})
        updated.append(item)
    if changed:
        _write_json(MODELS_FILE, updated)
    return changed


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
        if _safe_exists(path):
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


DOWNLOAD_STALL_TIMEOUT_SEC = int(os.environ.get("DOWNLOAD_STALL_TIMEOUT_SEC", "120"))
DOWNLOAD_AUTO_RETRY_MAX = int(os.environ.get("DOWNLOAD_AUTO_RETRY_MAX", "3"))
_last_disk_bytes: dict[str, tuple[int, float]] = {}


def _clamp_progress(value: Any) -> int:
    try:
        return max(0, min(100, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def reconcile_orphan_download_jobs(*, actor: str = "system") -> list[dict[str, Any]]:
    """プロセス再起動後などに残った queued/running ジョブを実態に合わせて修復する。

    - キャッシュ完了済み → completed
    - ワーカー不在かつ長時間未更新 → failed (DOWNLOAD_ORPHANED)
    - 完了済みでも progress > 100 ならクランプ
    """
    from app import download_worker

    now = time.time()
    actions: list[dict[str, Any]] = []
    idle_limit = max(DOWNLOAD_STALL_TIMEOUT_SEC * 2, 300)

    for job in load_jobs():
        model_id = str(job.get("model_id") or "")
        progress = job.get("progress")
        if job.get("status") == "completed" and progress is not None and int(progress or 0) > 100:
            job["progress"] = 100
            job["updated_at"] = now
            _save_job(job)
            actions.append({"job_id": job["id"], "model_id": model_id, "action": "clamped_progress"})

        if job.get("status") not in {"queued", "running"}:
            continue
        if not model_id:
            continue

        revision = _catalog_revision_for(model_id)
        if _cached_snapshot_path(model_id, revision) is not None:
            disk_bytes = _download_progress_bytes(model_id)
            total_bytes = int(job.get("total_bytes") or 0) or _repo_total_size_bytes(model_id, revision)
            job.update(
                status="completed",
                progress=100,
                downloaded_bytes=max(int(job.get("downloaded_bytes") or 0), disk_bytes),
                total_bytes=total_bytes or job.get("total_bytes") or 0,
                message="download completed",
                error=None,
                error_code=None,
                error_hint=None,
                updated_at=now,
            )
            _save_job(job)
            actions.append({"job_id": job["id"], "model_id": model_id, "action": "marked_completed"})
            continue

        worker_alive = download_worker.get_active_process(model_id) is not None
        updated_at = float(job.get("updated_at") or 0)
        idle_sec = now - updated_at if updated_at else idle_limit + 1
        if worker_alive or idle_sec < idle_limit:
            continue

        job.update(
            status="failed",
            error_code="DOWNLOAD_ORPHANED",
            error_hint="ダウンロードワーカーが停止したためジョブを終了しました。「続きから再開」で再試行できます。",
            message="download orphaned",
            updated_at=now,
        )
        _save_job(job)
        actions.append({"job_id": job["id"], "model_id": model_id, "action": "marked_orphaned", "idle_sec": int(idle_sec)})
    return actions


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


def _apply_cache_progress_to_job(job: dict[str, Any], model: dict[str, Any]) -> None:
    """既存キャッシュがあればジョブ進捗の初期値に反映する（再開時の巻き戻り防止）。"""
    model_id = str(model.get("id") or job.get("model_id") or "")
    if not model_id:
        return
    cached_bytes = _download_progress_bytes(model_id)
    total_bytes = _repo_total_size_bytes(model_id, model.get("revision"))
    if cached_bytes > 0:
        job["downloaded_bytes"] = cached_bytes
    if total_bytes > 0:
        job["total_bytes"] = total_bytes
    if job.get("downloaded_bytes") and job.get("total_bytes"):
        job["progress"] = _clamp_progress(
            int((int(job["downloaded_bytes"]) / int(job["total_bytes"])) * 100),
        )
    if cached_bytes > 0:
        job["message"] = "resuming from cache"


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
    if active and force:
        await cancel_active_download_jobs(model_id, actor=actor)
        active = []
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
    _apply_cache_progress_to_job(job, catalog[model_id])
    _save_job(job)
    await event_bus.publish("model_download", job, message="queued", actor=actor)
    asyncio.create_task(_run_download_job(job, catalog[model_id]))
    return job


async def _terminate_download_worker(model_id: str) -> None:
    await asyncio.to_thread(download_worker.terminate, model_id)


async def cancel_active_download_jobs(model_id: str, actor: str | None = None) -> dict[str, Any]:
    await _terminate_download_worker(model_id)
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


async def _sync_job_progress_from_disk(job: dict[str, Any]) -> bool:
    """ジョブ記録が古いがディスク上は進んでいる場合、進捗だけ同期する。"""
    model_id = str(job.get("model_id") or "")
    if not model_id:
        return False
    disk_bytes = _download_progress_bytes(model_id)
    recorded_bytes = int(job.get("downloaded_bytes") or 0)
    if disk_bytes <= recorded_bytes:
        return False
    total_bytes = int(job.get("total_bytes") or 0) or _repo_total_size_bytes(
        model_id, _catalog_revision_for(model_id)
    )
    progress = (
        _clamp_progress((disk_bytes / total_bytes) * 100)
        if total_bytes
        else _clamp_progress(job.get("progress"))
    )
    job.update(
        downloaded_bytes=disk_bytes,
        progress=progress,
        total_bytes=total_bytes or job.get("total_bytes") or 0,
        message="downloading",
        updated_at=time.time(),
    )
    _save_job(job)
    await event_bus.publish("model_download", job, message=job.get("message"), actor=job.get("actor"))
    return True


def _catalog_revision_for(model_id: str) -> str | None:
    for item in load_model_catalog():
        if item.get("id") == model_id:
            return item.get("revision")
    return None


async def resume_download_job(model_id: str, actor: str | None = None) -> dict[str, Any]:
    """実行中ジョブを止め、キャッシュ済みバイトから続きを再開する。"""
    latest_jobs = sorted(
        [item for item in load_jobs() if item.get("model_id") == model_id],
        key=lambda item: item.get("updated_at", 0),
        reverse=True,
    )
    latest = latest_jobs[0] if latest_jobs else None
    retry_count = int(latest.get("retry_count", 0)) if latest else 0
    parent_job_id = latest.get("id") if latest else None
    await _terminate_download_worker(model_id)
    await cancel_active_download_jobs(model_id, actor=actor)
    _last_disk_bytes.pop(model_id, None)
    return await start_download_job(
        model_id,
        actor=actor,
        force=True,
        retry_count=retry_count,
        parent_job_id=parent_job_id,
    )


async def inspect_stalled_download_jobs(*, actor: str = "download-watchdog") -> list[dict[str, Any]]:
    """進捗更新が止まったジョブを同期または自動再開する。"""
    now = time.time()
    actions: list[dict[str, Any]] = []
    seen_models: set[str] = set()

    for job in sorted(load_jobs(), key=lambda item: item.get("updated_at", 0)):
        if job.get("status") not in {"queued", "running"}:
            continue
        model_id = str(job.get("model_id") or "")
        if not model_id or model_id in seen_models:
            continue
        seen_models.add(model_id)

        disk_bytes = _download_progress_bytes(model_id)
        recorded_bytes = int(job.get("downloaded_bytes") or 0)
        if disk_bytes > recorded_bytes:
            await _sync_job_progress_from_disk(job)
            _last_disk_bytes[model_id] = (disk_bytes, now)
            actions.append(
                {
                    "model_id": model_id,
                    "job_id": job["id"],
                    "action": "synced_progress",
                    "downloaded_bytes": disk_bytes,
                }
            )
            continue

        prev = _last_disk_bytes.get(model_id)
        if prev is None:
            _last_disk_bytes[model_id] = (disk_bytes, now)
            continue
        prev_bytes, prev_ts = prev
        if disk_bytes != prev_bytes:
            _last_disk_bytes[model_id] = (disk_bytes, now)
            continue

        idle_sec = now - prev_ts
        if idle_sec < DOWNLOAD_STALL_TIMEOUT_SEC:
            continue

        retry_count = int(job.get("retry_count", 0))
        if retry_count >= DOWNLOAD_AUTO_RETRY_MAX:
            job.update(
                status="failed",
                error_code="DOWNLOAD_STALLED",
                error_hint="進捗が長時間止まったため自動再開の上限に達しました。手動で「続きから再開」を押してください。",
                message="download stalled",
                updated_at=now,
            )
            _save_job(job)
            await event_bus.publish("model_download", job, message=job.get("message"), actor=actor)
            actions.append(
                {
                    "model_id": model_id,
                    "job_id": job["id"],
                    "action": "marked_failed",
                    "idle_sec": int(idle_sec),
                }
            )
            _last_disk_bytes.pop(model_id, None)
            continue

        new_job = await start_download_job(
            model_id,
            actor=actor,
            force=True,
            retry_count=retry_count + 1,
            parent_job_id=job.get("id"),
        )
        _last_disk_bytes[model_id] = (disk_bytes, now)
        actions.append(
            {
                "model_id": model_id,
                "job_id": job["id"],
                "new_job_id": new_job["id"],
                "action": "auto_resumed",
                "idle_sec": int(idle_sec),
                "retry_count": retry_count + 1,
            }
        )
        await event_bus.publish(
            "model_download",
            new_job,
            message="進捗停止を検知したため、続きから自動再開しました",
            actor=actor,
        )
    return actions


async def _run_download_job(job: dict[str, Any], model: dict[str, Any]) -> None:
    model_id = model["id"]

    async def update(**changes: Any) -> bool:
        current = _job_by_id(job["id"])
        if current:
            job.update(current)
        if job.get("status") == "cancelled" and changes.get("status") != "cancelled":
            return False
        if "downloaded_bytes" in changes:
            changes["downloaded_bytes"] = max(
                int(changes["downloaded_bytes"] or 0),
                int(job.get("downloaded_bytes") or 0),
            )
        if "progress" in changes:
            changes["progress"] = _clamp_progress(
                max(
                    int(changes["progress"] or 0),
                    int(job.get("progress") or 0),
                )
            )
        job.update(changes)
        job["updated_at"] = time.time()
        _save_job(job)
        await event_bus.publish("model_download", job, message=job.get("message"), actor=job.get("actor"))
        return True

    await update(status="running", message="download started")

    cache_dir = os.environ.get("HF_HOME", "/app/hf-cache")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    disable_xet = os.environ.get("HF_HUB_DISABLE_XET", "1") == "1"
    spec: dict[str, Any] = {
        "repo_id": model_id,
        "cache_dir": cache_dir,
        "revision": model.get("revision"),
        "token": token,
        "disable_xet": disable_xet,
        "max_workers": int(os.environ.get("DOWNLOAD_WORKERS", "8")),
        "kind": "snapshot",
    }

    try:
        if model_id.lower().endswith("-gguf"):
            preferred = _preferred_gguf_file(model_id, model.get("revision"))
            if preferred:
                spec["kind"] = "gguf"
                spec["gguf_file"] = preferred
                await update(message=f"downloading selected GGUF file: {preferred}")
                total_bytes = _repo_file_size_bytes(model_id, preferred, model.get("revision"))
                if total_bytes > 0:
                    await update(total_bytes=total_bytes)
        else:
            total_bytes = _repo_total_size_bytes(model_id, model.get("revision"))
            if total_bytes > 0:
                await update(total_bytes=total_bytes)

        initial_size = _download_progress_bytes(model_id)
        if initial_size > 0:
            total_bytes = int(job.get("total_bytes") or 0)
            progress = _clamp_progress((initial_size / total_bytes) * 100) if total_bytes else 0
            await update(
                downloaded_bytes=initial_size,
                progress=progress,
                message="resuming from cache" if job.get("message") == "resuming from cache" else "downloading",
            )

        await _terminate_download_worker(model_id)
        if _is_cancelled(job["id"]):
            await update(status="cancelled", message="download cancelled by user")
            return
        proc = await asyncio.to_thread(download_worker.spawn, model_id, spec)
        last_size = int(job.get("downloaded_bytes") or 0)

        while proc.is_alive():
            if _is_cancelled(job["id"]):
                await _terminate_download_worker(model_id)
                await update(status="cancelled", message="download cancelled by user")
                return

            size = _download_progress_bytes(model_id)
            if size != last_size:
                total_bytes = int(job.get("total_bytes") or 0)
                progress = (
                    _clamp_progress((size / total_bytes) * 100)
                    if total_bytes
                    else _clamp_progress(job.get("progress"))
                )
                await update(
                    downloaded_bytes=size,
                    progress=progress,
                    message="downloading",
                )
                last_size = size
            await asyncio.sleep(2)

        download_worker.clear(model_id)
        if _is_cancelled(job["id"]) or (proc.exitcode is not None and proc.exitcode < 0):
            await update(status="cancelled", message="download cancelled by user")
            return
        if proc.exitcode != 0:
            # ワーカー異常終了でもキャッシュが揃っていれば完了扱い
            if _cached_snapshot_path(model_id, model.get("revision")) is not None:
                final_size = _download_progress_bytes(model_id)
                await update(status="completed", progress=100, message="download completed")
                if final_size > 0:
                    await update(downloaded_bytes=final_size)
                return
            raise RuntimeError(f"download worker exited with code {proc.exitcode}")

        final_size = _download_progress_bytes(model_id)
        await update(status="completed", progress=100, message="download completed")
        if final_size > 0:
            await update(downloaded_bytes=final_size)
    except DownloadCancelled:
        await _terminate_download_worker(model_id)
        await update(status="cancelled", message="download cancelled by user")
    except Exception as exc:
        download_worker.clear(model_id)
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
