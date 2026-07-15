"""Per-user LiteLLM API keys for the in-app chat UI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from app.auth import DATA_DIR
from app.litellm_client import litellm_request

CHAT_KEYS_FILE = DATA_DIR / "chat_keys.json"
# チャット UI から稼働中の各 vLLM モデル ID を直接指定できるようワイルドカードを許可する。
CHAT_KEY_MODELS = ["*"]
CHAT_KEY_ALIAS_PREFIX = "chat-ui-"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_chat_keys() -> dict[str, dict[str, Any]]:
    _ensure_data_dir()
    if not CHAT_KEYS_FILE.exists():
        return {}
    try:
        data = json.loads(CHAT_KEYS_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_chat_keys(data: dict[str, dict[str, Any]]) -> None:
    _ensure_data_dir()
    CHAT_KEYS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _chat_key_alias(username: str) -> str:
    return f"{CHAT_KEY_ALIAS_PREFIX}{username}"


def _extract_litellm_key(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("LiteLLM key generate returned invalid payload")
    for field in ("key", "token", "api_key"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("LiteLLM key generate response did not include a key")


def _key_info_holder(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    info = payload.get("info")
    return info if isinstance(info, dict) else payload


async def _find_litellm_key_hash_by_alias(alias: str) -> str | None:
    data = await litellm_request("GET", "/key/list")
    raw_keys = data.get("keys", []) if isinstance(data, dict) else data
    if not isinstance(raw_keys, list):
        return None
    for key_hash in raw_keys:
        if not isinstance(key_hash, str) or not key_hash.strip():
            continue
        info_payload = await litellm_request("GET", f"/key/info?key={quote(key_hash)}")
        info = _key_info_holder(info_payload)
        if info.get("key_alias") == alias:
            return key_hash
    return None


async def _delete_litellm_key_by_alias(alias: str) -> None:
    key_hash = await _find_litellm_key_hash_by_alias(alias)
    if not key_hash:
        return
    await litellm_request("POST", "/key/delete", {"keys": [key_hash]})


async def _issue_chat_key(user: dict[str, Any]) -> str:
    username = str(user.get("username") or "")
    if not username:
        raise ValueError("user username is required")

    alias = _chat_key_alias(username)
    payload: dict[str, Any] = {
        "user_id": user.get("litellm_user_id") or username,
        "team_id": user.get("litellm_team_id") or None,
        "models": list(CHAT_KEY_MODELS),
        "key_alias": alias,
    }
    try:
        data = await litellm_request("POST", "/key/generate", payload)
    except httpx.HTTPStatusError as exc:
        body = (exc.response.text or "").lower()
        if exc.response.status_code == 400 and "already exists" in body:
            await _delete_litellm_key_by_alias(alias)
            data = await litellm_request("POST", "/key/generate", payload)
        else:
            raise

    litellm_key = _extract_litellm_key(data)

    records = _load_chat_keys()
    records[username] = {
        "litellm_key": litellm_key,
        "key_alias": alias,
        "models": list(CHAT_KEY_MODELS),
        "created_at": time.time(),
    }
    _save_chat_keys(records)
    return litellm_key


async def get_or_create_chat_key(user: dict[str, Any]) -> str:
    username = str(user.get("username") or "")
    if not username:
        raise ValueError("user username is required")

    records = _load_chat_keys()
    entry = records.get(username)
    if isinstance(entry, dict):
        key = entry.get("litellm_key")
        stored_models = entry.get("models")
        if isinstance(key, str) and key.strip() and stored_models == list(CHAT_KEY_MODELS):
            return key.strip()
        if isinstance(key, str) and key.strip():
            # 旧キー（vllm-local のみ等）は再発行して稼働モデルを選べるようにする。
            return await regenerate_chat_key(user)

    return await _issue_chat_key(user)


async def regenerate_chat_key(user: dict[str, Any]) -> str:
    """Issue a fresh chat key (e.g. after LiteLLM revoked the previous one)."""
    username = str(user.get("username") or "")
    records = _load_chat_keys()
    records.pop(username, None)
    _save_chat_keys(records)
    await _delete_litellm_key_by_alias(_chat_key_alias(username))
    return await _issue_chat_key(user)


def chat_key_allows_wildcard(username: str) -> bool:
    entry = _load_chat_keys().get(username)
    return isinstance(entry, dict) and entry.get("models") == list(CHAT_KEY_MODELS)
