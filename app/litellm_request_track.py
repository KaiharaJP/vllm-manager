"""
LiteLLM 経由（識別ヘッダ付き）の /v1 リクエストを追跡し、トークン推定値を WebSocket で配信する。

既定構成では LiteLLM の model_list に extra_headers を足し、
バックエンドの OpenAI 互換プロキシが上流へ転送する際に SSE を観測する。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi.responses import Response, StreamingResponse
from starlette.requests import Request

from app.event_bus import event_bus
from app.request_history import append_record

LITELLM_SOURCE_HEADER = "x-vllm-manager-source"
LITELLM_SOURCE_VALUE = "litellm"

_PUBLISH_MIN_INTERVAL = 0.25
_MAX_MESSAGES_BYTES = 64 * 1024

_active: dict[str, dict[str, Any]] = {}
_lock = asyncio.Lock()
_last_publish_ts: dict[str, float] = {}


def header_marks_litellm(request: Request) -> bool:
    raw = request.headers.get(LITELLM_SOURCE_HEADER)
    return (raw or "").strip().lower() == LITELLM_SOURCE_VALUE


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("text"), str):
                    parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _extract_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt:
        return [{"role": "user", "content": prompt}]
    return []


def _extract_request_meta(payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "max_tokens": None,
            "message_count": 0,
            "prompt_char_est": 0,
            "request_summary": "",
            "_messages": None,
        }

    messages = _extract_messages(payload)
    char_est = 0
    last_user = ""
    for msg in messages:
        char_est += len(_content_text(msg.get("content")))
        role = str(msg.get("role") or "")
        if role == "user":
            last_user = _content_text(msg.get("content"))

    max_tokens = payload.get("max_tokens")
    try:
        max_tokens_int = int(max_tokens) if max_tokens is not None else None
    except (TypeError, ValueError):
        max_tokens_int = None

    preview = (last_user or _content_text(messages[-1].get("content") if messages else ""))[:120]
    preview = preview.replace("\n", " ").strip()
    summary = f"{len(messages)} msg, ~{char_est} chars"
    if max_tokens_int is not None:
        summary += f", max_tokens={max_tokens_int}"
    # プレビュー本文は WebSocket 向け公開行には含めない（request_summary は件数のみ）

    messages_store, truncated = _store_messages(messages)
    return {
        "max_tokens": max_tokens_int,
        "message_count": len(messages),
        "prompt_char_est": char_est,
        "request_summary": summary,
        "_preview": preview,
        "_messages": messages_store,
        "messages_truncated": truncated,
    }


def _store_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    if not messages:
        return [], False
    try:
        raw = json.dumps(messages, ensure_ascii=False)
    except (TypeError, ValueError):
        return messages[:1], True
    if len(raw.encode("utf-8")) <= _MAX_MESSAGES_BYTES:
        return messages, False
    # 先頭から収まる分だけ保持
    acc: list[dict[str, Any]] = []
    for msg in messages:
        trial = acc + [msg]
        try:
            if len(json.dumps(trial, ensure_ascii=False).encode("utf-8")) > _MAX_MESSAGES_BYTES:
                break
        except (TypeError, ValueError):
            break
        acc.append(msg)
    return acc or messages[:1], True


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def snapshot() -> list[dict[str, Any]]:
    return [_public_row(dict(v)) for v in _active.values()]


def get_active_detail(track_id: str) -> Optional[dict[str, Any]]:
    row = _active.get(track_id)
    if not row:
        return None
    out = _public_row(dict(row))
    messages = row.get("_messages")
    if messages is not None:
        out["messages"] = messages
        out["messages_truncated"] = bool(row.get("messages_truncated"))
    return out


def _new_row(
    *,
    track_id: str,
    subpath: str,
    model: str,
    stream: bool,
    request_payload: Optional[dict[str, Any]],
) -> dict[str, Any]:
    now = time.time()
    meta = _extract_request_meta(request_payload)
    return {
        "id": track_id,
        "endpoint": subpath,
        "model": model,
        "stream": stream,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "completion_chunks": 0,
        "status": "streaming" if stream else "pending",
        "phase": "prefill",
        "first_token_at": None,
        "prefill_tok_s": None,
        "gen_tok_s": None,
        "elapsed_s": 0.0,
        "error": None,
        "started_at": now,
        "updated_at": now,
        "max_tokens": meta["max_tokens"],
        "message_count": meta["message_count"],
        "prompt_char_est": meta["prompt_char_est"],
        "request_summary": meta["request_summary"],
        "_messages": meta["_messages"],
        "messages_truncated": meta.get("messages_truncated", False),
    }


def _mark_first_token(row: dict[str, Any], *, at: float | None = None) -> None:
    if row.get("first_token_at") is not None:
        return
    ts = at if at is not None else time.time()
    row["first_token_at"] = ts
    if row.get("phase") == "prefill":
        row["phase"] = "generate"
    started = float(row.get("started_at") or ts)
    pt = row.get("prompt_tokens")
    if isinstance(pt, int) and ts > started:
        row["prefill_tok_s"] = pt / (ts - started)


def _update_derived(row: dict[str, Any], *, finalize: bool = False) -> None:
    now = time.time()
    started = float(row.get("started_at") or now)
    row["elapsed_s"] = max(0.0, now - started)

    first_at = row.get("first_token_at")
    if isinstance(first_at, (int, float)) and row.get("prompt_tokens") is not None:
        pt = row["prompt_tokens"]
        if isinstance(pt, int) and first_at > started:
            row["prefill_tok_s"] = pt / (float(first_at) - started)

    ct = row.get("completion_tokens")
    if isinstance(first_at, (int, float)) and isinstance(ct, int) and ct > 0:
        gen_dt = now - float(first_at)
        if gen_dt > 0:
            row["gen_tok_s"] = ct / gen_dt

    if finalize and row.get("phase") != "error":
        row["phase"] = "done"


async def _finalize_row(row: dict[str, Any]) -> None:
    _update_derived(row, finalize=True)
    try:
        append_record(row)
    except Exception:
        pass


async def _register(track_id: str, row: dict[str, Any]) -> None:
    async with _lock:
        _active[track_id] = row


async def _unregister(track_id: str) -> None:
    async with _lock:
        _active.pop(track_id, None)
        _last_publish_ts.pop(track_id, None)


async def _publish_maybe(track_id: str, *, force: bool = False) -> None:
    async with _lock:
        row = _active.get(track_id)
        if not row:
            return
        now = time.time()
        last = _last_publish_ts.get(track_id, 0.0)
        if not force and (now - last) < _PUBLISH_MIN_INTERVAL:
            return
        _last_publish_ts[track_id] = now
        row["updated_at"] = now
        _update_derived(row)
        snap = _public_row(row)
    await event_bus.publish("litellm_proxy_request", snap)


def _extract_usage(obj: dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[int]]:
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    pt, ct, tt = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
    try:
        pti = int(pt) if pt is not None else None
    except (TypeError, ValueError):
        pti = None
    try:
        cti = int(ct) if ct is not None else None
    except (TypeError, ValueError):
        cti = None
    try:
        tti = int(tt) if tt is not None else None
    except (TypeError, ValueError):
        tti = None
    return pti, cti, tti


def _process_sse_data_line(row: dict[str, Any], line: bytes) -> None:
    if not line.startswith(b"data:"):
        return
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if not isinstance(obj, dict):
        return

    pt, ct, tt = _extract_usage(obj)
    if pt is not None:
        row["prompt_tokens"] = pt
    if ct is not None:
        row["completion_tokens"] = ct
        if ct > 0:
            _mark_first_token(row)
    if tt is not None:
        row["total_tokens"] = tt

    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    _mark_first_token(row)
                    row["completion_chunks"] = int(row.get("completion_chunks") or 0) + 1
            text = first.get("text")
            if isinstance(text, str) and text:
                _mark_first_token(row)
                row["completion_chunks"] = int(row.get("completion_chunks") or 0) + 1
            finish_reason = first.get("finish_reason")
            if finish_reason is not None and row.get("phase") == "generate":
                row["phase"] = "done"


def _response_headers_from_upstream(upstream: httpx.Response) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in upstream.headers.items():
        lk = key.lower()
        if lk in {"content-length", "transfer-encoding", "connection", "content-encoding"}:
            continue
        out[key] = value
    return out


async def proxy_litellm_tracked_v1(
    *,
    subpath: str,
    target: str,
    body: bytes,
    forward_headers: dict[str, str],
    timeout: httpx.Timeout,
    request_payload: Optional[dict[str, Any]],
) -> Response:
    track_id = str(uuid.uuid4())
    model = ""
    stream = False
    if isinstance(request_payload, dict):
        model = str(request_payload.get("model") or "")
        stream = bool(request_payload.get("stream"))

    row = _new_row(
        track_id=track_id,
        subpath=subpath,
        model=model,
        stream=stream,
        request_payload=request_payload,
    )
    await _register(track_id, row)
    await _publish_maybe(track_id, force=True)

    client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    try:
        if stream:
            return await _stream_tracked(client, target, body, forward_headers, track_id, row)
        return await _buffer_tracked(client, target, body, forward_headers, track_id, row)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        row["phase"] = "error"
        await _finalize_row(row)
        await _publish_maybe(track_id, force=True)
        await _unregister(track_id)
        await client.aclose()
        raise


async def _buffer_tracked(
    client: httpx.AsyncClient,
    target: str,
    body: bytes,
    forward_headers: dict[str, str],
    track_id: str,
    row: dict[str, Any],
) -> Response:
    try:
        upstream = await client.request("POST", target, content=body, headers=forward_headers)
        content = await upstream.aread()
        ct = upstream.headers.get("content-type", "application/json")
        headers = _response_headers_from_upstream(upstream)

        if upstream.status_code == 200:
            try:
                obj = json.loads(content.decode("utf-8"))
                if isinstance(obj, dict):
                    pt, cpt, tt = _extract_usage(obj)
                    if pt is not None:
                        row["prompt_tokens"] = pt
                    if cpt is not None:
                        row["completion_tokens"] = cpt
                    if tt is not None:
                        row["total_tokens"] = tt
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            _mark_first_token(row)
            row["status"] = "completed"
            await _finalize_row(row)
        else:
            row["status"] = "error"
            row["error"] = f"HTTP {upstream.status_code}"
            row["phase"] = "error"
            await _finalize_row(row)

        await _publish_maybe(track_id, force=True)
        return Response(content=content, status_code=upstream.status_code, headers=headers, media_type=ct)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        row["phase"] = "error"
        await _finalize_row(row)
        await _publish_maybe(track_id, force=True)
        raise
    finally:
        await _unregister(track_id)
        await client.aclose()


async def _stream_tracked(
    client: httpx.AsyncClient,
    target: str,
    body: bytes,
    forward_headers: dict[str, str],
    track_id: str,
    row: dict[str, Any],
) -> StreamingResponse:
    req = client.build_request("POST", target, headers=forward_headers, content=body)
    resp = await client.send(req, stream=True)

    media_type = resp.headers.get("content-type", "text/event-stream")
    out_headers = _response_headers_from_upstream(resp)

    if resp.status_code != 200:
        body_bytes = await resp.aread()
        row["status"] = "error"
        row["error"] = f"HTTP {resp.status_code}"
        row["phase"] = "error"
        await _finalize_row(row)
        await _publish_maybe(track_id, force=True)
        await _unregister(track_id)
        await resp.aclose()
        await client.aclose()
        return Response(content=body_bytes, status_code=resp.status_code, headers=out_headers, media_type=media_type)

    async def body_iter() -> AsyncIterator[bytes]:
        carry = b""
        try:
            async for chunk in resp.aiter_bytes():
                carry += chunk
                while b"\n" in carry:
                    line, carry = carry.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    if line.startswith(b"data:"):
                        _process_sse_data_line(row, line)
                        await _publish_maybe(track_id)
                yield chunk
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            row["phase"] = "error"
            await _finalize_row(row)
            await _publish_maybe(track_id, force=True)
            raise
        finally:
            try:
                if carry.strip():
                    line = carry.rstrip(b"\r\n")
                    if line.startswith(b"data:"):
                        _process_sse_data_line(row, line)
                if row.get("status") != "error":
                    row["status"] = "completed"
                    await _finalize_row(row)
                await _publish_maybe(track_id, force=True)
            finally:
                await resp.aclose()
                await _unregister(track_id)
                await client.aclose()

    return StreamingResponse(body_iter(), status_code=200, headers=out_headers, media_type=media_type)
