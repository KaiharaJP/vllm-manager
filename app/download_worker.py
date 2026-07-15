"""Terminable child-process helpers for Hugging Face downloads."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Any

_active_processes: dict[str, mp.Process] = {}


def run_download_task(spec: dict[str, Any]) -> int:
    """Run snapshot or single-file download in an isolated process."""
    if spec.get("disable_xet"):
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    cache_dir = spec.get("cache_dir") or os.environ.get("HF_HOME", "/app/hf-cache")
    token = spec.get("token")
    repo_id = spec["repo_id"]
    revision = spec.get("revision")

    try:
        if spec.get("kind") == "gguf":
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=repo_id,
                filename=spec["gguf_file"],
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                force_download=False,
            )
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                max_workers=int(spec.get("max_workers") or 8),
            )
        return 0
    except Exception:
        return 1


def _child_entry(spec: dict[str, Any]) -> None:
    sys.exit(run_download_task(spec))


def spawn(model_id: str, spec: dict[str, Any]) -> mp.Process:
    """Start a download worker for model_id and track it."""
    terminate(model_id, join_timeout=0)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_child_entry,
        args=(spec,),
        name=f"hf-download-{model_id}",
        daemon=False,
    )
    proc.start()
    _active_processes[model_id] = proc
    return proc


def get_active_process(model_id: str) -> mp.Process | None:
    proc = _active_processes.get(model_id)
    if proc is None:
        return None
    if not proc.is_alive():
        _active_processes.pop(model_id, None)
        return None
    return proc


def terminate(model_id: str, *, join_timeout: float = 10) -> bool:
    """Terminate a running worker. Returns True if a process was stopped."""
    proc = _active_processes.pop(model_id, None)
    if proc is None:
        return False
    if proc.is_alive():
        proc.terminate()
        proc.join(join_timeout)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
    return True


def clear(model_id: str) -> None:
    _active_processes.pop(model_id, None)


def active_model_ids() -> list[str]:
    return [model_id for model_id, proc in list(_active_processes.items()) if proc.is_alive()]
