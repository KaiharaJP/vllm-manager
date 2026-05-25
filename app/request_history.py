"""
LiteLLM 経由リクエストの完了履歴を JSONL で永続化する。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(os.environ.get("VLLM_MANAGER_DATA_DIR", "/tmp/vllm-manager-data"))
HISTORY_FILE = DATA_DIR / "request_history.jsonl"

MAX_ENTRIES = 500
MAX_AGE_SEC = 7 * 24 * 3600
MAX_MESSAGES_BYTES = 64 * 1024


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _truncate_messages(messages: Any) -> tuple[Any, bool]:
    if messages is None:
        return None, False
    try:
        raw = json.dumps(messages, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(messages)[:MAX_MESSAGES_BYTES], True
    if len(raw.encode("utf-8")) <= MAX_MESSAGES_BYTES:
        return messages, False
    return json.loads(raw[:MAX_MESSAGES_BYTES] + '"}'), True


def row_to_history_record(row: dict[str, Any]) -> dict[str, Any]:
    """追跡行から履歴レコードを構築する。"""
    now = time.time()
    messages = row.get("_messages")
    messages_out, truncated = _truncate_messages(messages)
    return {
        "id": row.get("id"),
        "started_at": row.get("started_at"),
        "completed_at": now,
        "endpoint": row.get("endpoint"),
        "model": row.get("model"),
        "stream": row.get("stream"),
        "max_tokens": row.get("max_tokens"),
        "message_count": row.get("message_count"),
        "prompt_char_est": row.get("prompt_char_est"),
        "request_summary": row.get("request_summary"),
        "messages": messages_out,
        "messages_truncated": truncated,
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "total_tokens": row.get("total_tokens"),
        "status": row.get("status"),
        "phase": row.get("phase"),
        "error": row.get("error"),
        "prefill_tok_s": row.get("prefill_tok_s"),
        "gen_tok_s": row.get("gen_tok_s"),
        "elapsed_s": row.get("elapsed_s"),
    }


def append_record(row: dict[str, Any]) -> None:
    record = row_to_history_record(row)
    _ensure_data_dir()
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _prune_file()


def _prune_file() -> None:
    if not HISTORY_FILE.exists():
        return
    try:
        lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    now = time.time()
    kept: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            completed = float(obj.get("completed_at") or 0)
            if completed > 0 and (now - completed) > MAX_AGE_SEC:
                continue
            kept.append(line)
        except json.JSONDecodeError:
            continue
    if len(kept) > MAX_ENTRIES:
        kept = kept[-MAX_ENTRIES:]
    try:
        HISTORY_FILE.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    except OSError:
        pass


def list_records(*, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    if not HISTORY_FILE.exists():
        return [], 0
    try:
        lines = [ln for ln in HISTORY_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return [], 0
    records: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    total = len(records)
    end = offset + limit
    return records[offset:end], total


def get_record(record_id: str) -> Optional[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return None
    try:
        lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("id") == record_id:
                return obj
        except json.JSONDecodeError:
            continue
    return None
