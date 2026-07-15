"""Persistent LiteLLM inference keys for CLI / agent automation (get-or-create)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from app.auth import DATA_DIR
from app.litellm_client import litellm_request

CLI_KEYS_FILE = DATA_DIR / "cli_inference_keys.json"
DEFAULT_KEY_ALIAS = "vllm-cli"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_cli_keys() -> dict[str, dict[str, Any]]:
    _ensure_data_dir()
    if not CLI_KEYS_FILE.exists():
        return {}
    try:
        data = json.loads(CLI_KEYS_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cli_keys(data: dict[str, dict[str, Any]]) -> None:
    _ensure_data_dir()
    CLI_KEYS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _normalize_models(models: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for model in models:
        value = str(model).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned or ["*"]


def _models_equal(a: Any, b: list[str]) -> bool:
    if not isinstance(a, list):
        return False
    left = _normalize_models([str(x) for x in a])
    right = _normalize_models(b)
    return sorted(left) == sorted(right)


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


def _record_key(username: str) -> str:
    return username


async def _issue_cli_key(user: dict[str, Any], models: list[str], key_alias: str) -> str:
    username = str(user.get("username") or "")
    if not username:
        raise ValueError("user username is required")
    models = _normalize_models(models)

    payload: dict[str, Any] = {
        "user_id": user.get("litellm_user_id") or username,
        "team_id": user.get("litellm_team_id") or None,
        "models": models,
        "key_alias": key_alias,
    }
    try:
        data = await litellm_request("POST", "/key/generate", payload)
    except httpx.HTTPStatusError as exc:
        body = (exc.response.text or "").lower()
        if exc.response.status_code == 400 and "already exists" in body:
            await _delete_litellm_key_by_alias(key_alias)
            data = await litellm_request("POST", "/key/generate", payload)
        else:
            raise

    litellm_key = _extract_litellm_key(data)
    records = _load_cli_keys()
    records[_record_key(username)] = {
        "litellm_key": litellm_key,
        "key_alias": key_alias,
        "models": models,
        "created_at": time.time(),
    }
    _save_cli_keys(records)
    return litellm_key


async def ensure_cli_inference_key(
    user: dict[str, Any],
    *,
    models: list[str] | None = None,
    key_alias: str = DEFAULT_KEY_ALIAS,
    force: bool = False,
) -> dict[str, Any]:
    """Return existing CLI automation key or mint one (same models + alias)."""
    username = str(user.get("username") or "")
    if not username:
        raise ValueError("user username is required")
    models = _normalize_models(models or ["*"])
    alias = (key_alias or DEFAULT_KEY_ALIAS).strip() or DEFAULT_KEY_ALIAS

    records = _load_cli_keys()
    entry = records.get(_record_key(username))
    if (
        not force
        and isinstance(entry, dict)
        and entry.get("key_alias") == alias
        and _models_equal(entry.get("models"), models)
    ):
        key = entry.get("litellm_key")
        if isinstance(key, str) and key.strip():
            return {
                "key": key.strip(),
                "reused": True,
                "key_alias": alias,
                "models": models,
            }

    if force or isinstance(entry, dict):
        await _delete_litellm_key_by_alias(alias)
        records = _load_cli_keys()
        records.pop(_record_key(username), None)
        _save_cli_keys(records)

    key = await _issue_cli_key(user, models, alias)
    return {
        "key": key,
        "reused": False,
        "key_alias": alias,
        "models": models,
    }
