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


async def status() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{_base_url()}/health")
        return {"healthy": response.status_code == 200, "url": _base_url()}
    except Exception:
        return {"healthy": False, "url": _base_url()}
