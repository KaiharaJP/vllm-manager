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

LITELLM_SOURCE_HEADER = "x-vllm-manager-source"
LITELLM_SOURCE_VALUE = "litellm"

_PUBLISH_MIN_INTERVAL = 0.25

_active: dict[str, dict[str, Any]] = {}
_lock = asyncio.Lock()
_last_publish_ts: dict[str, float] = {}


def header_marks_litellm(request: Request) -> bool:
    raw = request.headers.get(LITELLM_SOURCE_HEADER)
    return (raw or "").strip().lower() == LITELLM_SOURCE_VALUE


def snapshot() -> list[dict[str, Any]]:
    return [dict(v) for v in _active.values()]


def _new_row(
    *,
    track_id: str,
    subpath: str,
    model: str,
    stream: bool,
) -> dict[str, Any]:
    now = time.time()
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
        snap = dict(row)
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

    row = _new_row(track_id=track_id, subpath=subpath, model=model, stream=stream)
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
            _update_derived(row, finalize=True)
        else:
            row["status"] = "error"
            row["error"] = f"HTTP {upstream.status_code}"
            row["phase"] = "error"

        await _publish_maybe(track_id, force=True)
        return Response(content=content, status_code=upstream.status_code, headers=headers, media_type=ct)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
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
                    _update_derived(row, finalize=True)
                await _publish_maybe(track_id, force=True)
            finally:
                await resp.aclose()
                await _unregister(track_id)
                await client.aclose()

    return StreamingResponse(body_iter(), status_code=200, headers=out_headers, media_type=media_type)
