"""Append-only audit log persisted under VLLM_MANAGER_DATA_DIR."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(os.environ.get("VLLM_MANAGER_DATA_DIR", "/tmp/vllm-manager-data"))
AUDIT_LOG_FILE = DATA_DIR / "audit.log"

# 監査ログが無制限に肥大化しないようにするためのローテーション設定。
MAX_AUDIT_LOG_BYTES = int(
    os.environ.get("VLLM_MANAGER_AUDIT_LOG_MAX_BYTES", str(20 * 1024 * 1024))
)
AUDIT_LOG_BACKUP_COUNT = int(os.environ.get("VLLM_MANAGER_AUDIT_LOG_BACKUPS", "3"))

# WebSocket / 一般向けイベントから除外する監査対象外タイプ
METRICS_EVENT_TYPES = frozenset(
    {
        "metrics",
        "metrics_scrape_error",
        "litellm_proxy_request",
        "pong",
    }
)


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _rotate_if_needed() -> None:
    """audit.log が上限サイズを超えたら世代ローテーションする（古い世代は削除）。"""
    try:
        if not AUDIT_LOG_FILE.exists() or AUDIT_LOG_FILE.stat().st_size < MAX_AUDIT_LOG_BYTES:
            return
    except OSError:
        return

    for idx in range(AUDIT_LOG_BACKUP_COUNT - 1, 0, -1):
        src = DATA_DIR / f"audit.log.{idx}"
        dst = DATA_DIR / f"audit.log.{idx + 1}"
        if src.exists():
            try:
                src.replace(dst)
            except OSError:
                pass
    try:
        AUDIT_LOG_FILE.replace(DATA_DIR / "audit.log.1")
    except OSError:
        pass

    oldest = DATA_DIR / f"audit.log.{AUDIT_LOG_BACKUP_COUNT + 1}"
    if oldest.exists():
        try:
            oldest.unlink()
        except OSError:
            pass


def append_audit(
    *,
    action: str,
    actor: Optional[str] = None,
    message: Optional[str] = None,
    data: Any = None,
) -> dict[str, Any]:
    """Write one JSON line to audit.log."""
    _ensure_data_dir()
    _rotate_if_needed()
    entry = {
        "timestamp": time.time(),
        "action": action,
        "actor": actor,
        "message": message,
        "data": data,
    }
    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def should_audit_event(event_type: str) -> bool:
    return event_type not in METRICS_EVENT_TYPES


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def read_recent(limit: int = 100) -> list[dict[str, Any]]:
    """Return the most recent audit entries (best effort).

    ローテーション直後で現行ファイルの件数が足りない場合は、直近の
    バックアップ世代（audit.log.1）からも補完する。
    """
    lines = _read_lines(AUDIT_LOG_FILE)
    if len(lines) < limit:
        lines = _read_lines(DATA_DIR / "audit.log.1") + lines
    entries: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            entries.append(item)
    return entries
