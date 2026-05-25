"""HTTP のみのサービスヘルスチェック（vLLM / LiteLLM 推論プローブは使わない）。"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.litellm_client import liveliness, readiness
from app.server_manager import get_status

LITELLM_INFERENCE_PROBE_PROMPT = "test from litellm"


async def check_all_services() -> dict[str, Any]:
    """手動ヘルスチェック用。キューに載せず各サービスの HTTP 応答だけ確認する。"""
    checked_at = time.time()
    vllm_status = get_status()
    vllm_message = _vllm_message(vllm_status)

    litellm_live = await liveliness()
    litellm_ready = await readiness()

    return {
        "checked_at": checked_at,
        "method": "http",
        "vllm": {
            "healthy": bool(vllm_status.get("healthy")),
            "running": bool(vllm_status.get("running")),
            "port": int(vllm_status.get("vllm_port") or 8001),
            "model": vllm_status.get("model"),
            "message": vllm_message,
        },
        "litellm": {
            "liveliness": bool(litellm_live.get("healthy")),
            "readiness": bool(litellm_ready.get("healthy")),
            "url": litellm_live.get("url"),
            "liveliness_detail": litellm_live.get("detail"),
            "readiness_detail": litellm_ready.get("detail"),
        },
        "backend": {"healthy": True, "message": "API 応答あり"},
    }


def _vllm_message(status: dict[str, Any]) -> str:
    if status.get("healthy"):
        model = status.get("model")
        return f"vLLM /health OK（port {status.get('vllm_port')}）" + (f" — {model}" if model else "")
    if status.get("running"):
        return f"プロセスは動作中だが /health 未応答（port {status.get('vllm_port')}）"
    return "vLLM 未起動"


def is_inference_health_probe(payload: dict[str, Any] | None) -> bool:
    """LiteLLM の推論ベースモデル点検（test from litellm）かどうか。"""
    if not isinstance(payload, dict):
        return False
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip() == LITELLM_INFERENCE_PROBE_PROMPT:
        return True
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return False
    msg = messages[0]
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, str) or content.strip() != LITELLM_INFERENCE_PROBE_PROMPT:
        return False
    max_tokens = payload.get("max_tokens")
    try:
        if max_tokens is not None and int(max_tokens) <= 10:
            return True
    except (TypeError, ValueError):
        pass
    return False
