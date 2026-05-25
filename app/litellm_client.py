"""Thin LiteLLM Proxy admin API client."""

import os
from typing import Any

import httpx


def _base_url() -> str:
    return os.environ.get("LITELLM_INTERNAL_URL", "http://litellm:4000").rstrip("/")


def _master_key() -> str:
    return os.environ.get("LITELLM_MASTER_KEY", "sk-vllm-default-key")


async def litellm_request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    headers = {"Authorization": f"Bearer {_master_key()}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.request(method, f"{_base_url()}{path}", headers=headers, json=payload)
        response.raise_for_status()
        if response.content:
            return response.json()
        return {"success": True}


async def liveliness() -> dict[str, Any]:
    """LiteLLM プロセス生存確認（推論・キューに載せない）。"""
    url = f"{_base_url()}/health/liveliness"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url)
        body = (response.text or "").strip()[:200]
        return {
            "healthy": response.status_code == 200,
            "url": _base_url(),
            "detail": body or None,
        }
    except Exception as exc:
        return {"healthy": False, "url": _base_url(), "detail": str(exc)}


async def readiness() -> dict[str, Any]:
    """LiteLLM がリクエスト受付可能か（DB 等・推論なし）。"""
    url = f"{_base_url()}/health/readiness"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
        detail: Any = None
        if response.content:
            try:
                detail = response.json()
            except Exception:
                detail = (response.text or "")[:200]
        healthy = response.status_code == 200
        if isinstance(detail, dict):
            db = detail.get("db")
            status = str(detail.get("status") or "").lower()
            if db == "Not connected":
                healthy = False
            elif status not in ("connected", "healthy", ""):
                healthy = False
        return {"healthy": healthy, "url": _base_url(), "detail": detail}
    except Exception as exc:
        return {"healthy": False, "url": _base_url(), "detail": str(exc)}


async def status() -> dict[str, Any]:
    """後方互換: 推論 HC ではなく liveliness のみ。"""
    live = await liveliness()
    return {
        "healthy": bool(live.get("healthy")),
        "url": live.get("url") or _base_url(),
        "check": "liveliness",
    }
