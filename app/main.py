"""
vLLM Manager - FastAPI アプリケーション

API エンドポイント + WebSocket メトリクス配信を提供する。
"""

import os
import json
from urllib.parse import quote, urlencode
from contextlib import asynccontextmanager, suppress
from typing import Any, Optional

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, APIRouter, HTTPException, Request, Response, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx

from app.server_manager import (
    get_status,
    start_server,
    stop_server,
    stop_instance,
    stop_server_by_pid,
    restart_server,
    get_log_lines,
    get_context_presets,
    list_running_servers,
    list_instances,
    load_config,
    restore_managed_instances,
    run_smoke_test,
)
from app.system_metrics import get_system_metrics
from app.auth import (
    authenticate,
    check_login_allowed,
    collect_security_warnings,
    create_api_key,
    create_session,
    get_current_user,
    get_user_from_token,
    list_api_keys,
    load_users,
    public_user,
    record_login_failure,
    record_login_success,
    require_admin,
    revoke_api_key,
    upsert_user,
    update_user,
    user_must_change_password,
)
from app.audit_log import read_recent as read_audit_log
from app.event_bus import event_bus
from app.litellm_client import litellm_request, status as litellm_status
from app.chat_keys import chat_key_allows_wildcard, get_or_create_chat_key, regenerate_chat_key
from app.cli_inference_keys import DEFAULT_KEY_ALIAS, ensure_cli_inference_key
from app.model_manager import (
    cancel_active_download_jobs,
    delete_model,
    delete_model_cache,
    inspect_stalled_download_jobs,
    load_jobs,
    load_model_catalog,
    migrate_misclassified_rerank_models,
    reconcile_orphan_download_jobs,
    resume_download_job,
    save_model,
    start_download_job,
)
from app.metrics_scraper import MetricsScraper
from app.health_watchdog import HealthWatchdog
from app.resource_watchdog import ResourceWatchdog
from app.download_watchdog import DownloadWatchdog
from app.litellm_request_track import (
    get_active_detail,
    header_marks_litellm,
    proxy_litellm_tracked_v1,
    snapshot as litellm_proxy_snapshot,
)
from app.service_health import check_all_services, is_inference_health_probe
from app import storage_info, training_manager
from app.request_history import get_record as get_history_record, list_records as list_history_records

BACKEND_INTERNAL_PORT = int(os.environ.get("BACKEND_INTERNAL_PORT", "8000"))


def _proxy_upstream_read_timeout_sec() -> float:
    raw = os.environ.get("PROXY_UPSTREAM_READ_TIMEOUT_SEC", "600")
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0


def _proxy_upstream_timeout() -> httpx.Timeout:
    return httpx.Timeout(_proxy_upstream_read_timeout_sec(), connect=10.0)


def _env_flag(name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_MM_BLOCK_TYPES = frozenset({"image_url", "input_image", "image", "video_url", "input_video", "video"})


def _content_has_multimodal(content: Any) -> bool:
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and str(part.get("type") or "") in _MM_BLOCK_TYPES:
                return True
            if isinstance(part, dict) and _normalize_image_url_block(part):
                return True
    return False


def _payload_has_multimodal(payload: Optional[dict[str, Any]]) -> bool:
    """chat/completions または responses 相当の payload に画像・動画ブロックがあるか。"""
    if not isinstance(payload, dict):
        return False
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and _content_has_multimodal(msg.get("content")):
                return True
    input_value = payload.get("input")
    if isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") == "message" and _content_has_multimodal(item.get("content")):
                return True
            if _normalize_image_url_block(item):
                return True
    return False


def _should_force_stream(
    request: Request,
    cfg: dict[str, Any],
    *,
    request_payload: Optional[dict[str, Any]] = None,
) -> bool:
    """LiteLLM 経由の chat/completions で stream=false を stream=true に上書きするか。"""
    if not header_marks_litellm(request):
        return False
    if not _env_flag("PROXY_FORCE_STREAM", default=True):
        return False
    if not bool(cfg.get("force_stream", True)):
        return False
    if _payload_has_multimodal(request_payload):
        return False
    return True


def _is_backend_self_port(port: Any) -> bool:
    try:
        return int(port) == BACKEND_INTERNAL_PORT
    except Exception:
        return False


# --- Pydantic モデル ---

class ServerStartRequest(BaseModel):
    model_id: str
    context_length: Optional[int] = Field(default=None, ge=1024, le=262144)
    max_num_seqs: int = Field(default=6, ge=1, le=20)
    default_max_tokens: int = Field(default=512, ge=1, le=262144)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    default_frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    default_presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    gpu_memory_mode: str = Field(default="auto", pattern="^(auto|manual)$")
    gpu_memory_utilization: float = Field(default=0.85, ge=0.1, le=1.0)
    tensor_parallel_size: int = Field(default=1, ge=1, le=8)
    gpu_devices: str = "all"
    speculative_config: dict[str, Any] = Field(default_factory=dict)
    download_model: bool = True
    enable_auto_tool_choice: bool = False
    tool_call_parser: str = ""
    force_stream: bool = True
    enable_lora: bool = False
    max_lora_rank: Optional[int] = Field(default=None, ge=8, le=512)
    limit_mm_per_prompt: Optional[dict[str, int]] = None
    mm_encoder_tp_mode: str = ""
    mm_processor_cache_type: str = ""
    task_type: Optional[str] = Field(default=None, pattern="^(chat|embedding|rerank)$")
    trust_remote_code: Optional[bool] = None
    instance_id: Optional[str] = None
    instance_name: Optional[str] = None
    create_new_instance: bool = False


def _resolve_start_request(req: ServerStartRequest, catalog_entry: dict[str, Any]) -> dict[str, Any]:
    """HTTP 起動リクエストをカタログ情報とマージして start_server 向けに正規化する。"""
    task_type = req.task_type or catalog_entry.get("task_type") or "chat"
    context_length = req.context_length
    if context_length is None:
        context_length = catalog_entry.get("recommended_context_length")
    if context_length is None:
        context_length = 8192 if task_type in {"embedding", "rerank"} else 131072
    trust_remote_code = (
        req.trust_remote_code
        if req.trust_remote_code is not None
        else bool(catalog_entry.get("trust_remote_code"))
    )
    create_new_instance = req.create_new_instance or task_type in {"embedding", "rerank"}
    return {
        "task_type": task_type,
        "context_length": int(context_length),
        "trust_remote_code": trust_remote_code,
        "create_new_instance": create_new_instance,
    }


class StopInstanceRequest(BaseModel):
    instance_id: str = Field(min_length=1, max_length=64)


class ServerStatusResponse(BaseModel):
    running: bool
    healthy: bool
    pid: Optional[int] = None
    vllm_port: int
    model: Optional[str] = None
    uptime_seconds: float


class ApiResponse(BaseModel):
    success: bool
    message: str
    steps: list[str] = Field(default_factory=list)


class StopServerByPidRequest(BaseModel):
    pid: int = Field(ge=1)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = Field(default="user", pattern="^(admin|user)$")
    litellm_user_id: Optional[str] = None
    litellm_team_id: Optional[str] = None


class UserUpdateRequest(BaseModel):
    password: Optional[str] = None
    role: Optional[str] = Field(default=None, pattern="^(admin|user)$")
    litellm_user_id: Optional[str] = None
    litellm_team_id: Optional[str] = None
    disabled: Optional[bool] = None


class SelfUserUpdateRequest(BaseModel):
    password: Optional[str] = None
    litellm_team_id: Optional[str] = None


class SelfApiKeyRequest(BaseModel):
    models: list[str] = Field(..., min_length=1)
    key_alias: Optional[str] = Field(default=None, max_length=128)
    max_budget: Optional[float] = None
    budget_duration: Optional[str] = None
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None


class SelfApiKeyEnsureRequest(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["*"], min_length=1)
    key_alias: str = Field(default=DEFAULT_KEY_ALIAS, min_length=1, max_length=128)
    force: bool = False


class SelfTokenCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


class AdminUserApiKeyRequest(BaseModel):
    models: list[str] = Field(..., min_length=1)
    key_alias: Optional[str] = Field(default=None, max_length=128)
    max_budget: Optional[float] = None
    budget_duration: Optional[str] = None
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None


class ChatUiMessage(BaseModel):
    role: str
    content: str


class ChatUiCompletionRequest(BaseModel):
    model: str
    messages: list[ChatUiMessage] = Field(..., min_length=1)
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


async def _litellm_list_keys_detailed() -> list[dict[str, Any]]:
    """Return detailed key info entries via /key/list + /key/info."""
    data = await litellm_request("GET", "/key/list")
    raw_keys = data.get("keys", []) if isinstance(data, dict) else data
    if not isinstance(raw_keys, list):
        return []

    detailed: list[dict[str, Any]] = []
    for k in raw_keys:
        if not isinstance(k, str):
            continue
        info = await litellm_request("GET", f"/key/info?key={quote(k)}")
        if isinstance(info, dict):
            detailed.append(info)
    return detailed


def _sanitize_models(models: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for model in models:
        if not isinstance(model, str):
            continue
        value = model.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    if not cleaned:
        raise HTTPException(status_code=400, detail="models must contain at least one non-empty model id")
    return cleaned


class ModelRegisterRequest(BaseModel):
    id: str
    name: Optional[str] = None
    size: Optional[str] = None
    revision: Optional[str] = None
    gated: bool = False
    trust_remote_code: bool = False
    recommended_context_length: int = Field(default=8192, ge=1024)
    required_gpu_memory_gb: Optional[float] = None
    output_dimension: Optional[int] = Field(default=None, ge=1, le=65536)
    license_note: Optional[str] = Field(default=None, max_length=256)
    allowed_roles: list[str] = Field(default_factory=lambda: ["admin", "user"])
    task_type: str = Field(default="chat", pattern="^(chat|embedding|rerank)$")


class DownloadRequest(BaseModel):
    model_id: str
    force: bool = False


class LiteLLMProxyRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


# --- グローバル状態 ---
metrics_scraper: Optional[MetricsScraper] = None
health_watchdog: Optional[HealthWatchdog] = None
resource_watchdog: Optional[ResourceWatchdog] = None
download_watchdog: Optional[DownloadWatchdog] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # スタートアップ
    global metrics_scraper, health_watchdog, resource_watchdog, download_watchdog
    metrics_scraper = MetricsScraper(
        vllm_metrics_url=f"http://localhost:{os.environ.get('VLLM_PORT', '8001')}/metrics",
        scrape_interval=5.0,
        event_publisher=event_bus.publish,
    )
    await metrics_scraper.start()

    restore_results = restore_managed_instances()
    if restore_results:
        await event_bus.publish(
            "server_restore",
            {"results": restore_results},
            message=f"Restored {sum(1 for r in restore_results if r.get('success'))}/{len(restore_results)} vLLM instance(s)",
            actor="system",
        )

    health_watchdog = HealthWatchdog(
        check_interval=30.0,
        event_publisher=event_bus.publish,
        smoke_test_runner=run_smoke_test,
    )
    await health_watchdog.start()

    resource_watchdog = ResourceWatchdog(event_publisher=event_bus.publish)
    await resource_watchdog.start()

    download_watchdog = DownloadWatchdog(inspector=inspect_stalled_download_jobs)
    await download_watchdog.start()

    orphan_actions = reconcile_orphan_download_jobs(actor="system")
    if orphan_actions:
        await event_bus.publish(
            "model_download",
            {"actions": orphan_actions},
            message=f"Reconciled {len(orphan_actions)} orphan download job(s)",
            actor="system",
        )

    rerank_migrations = migrate_misclassified_rerank_models()
    if rerank_migrations:
        await event_bus.publish(
            "model_registered",
            {"migrations": rerank_migrations},
            message=f"Migrated {len(rerank_migrations)} model(s) from embedding to rerank",
            actor="system",
        )

    yield

    # シャットダウン
    if metrics_scraper:
        await metrics_scraper.stop()
    if health_watchdog:
        await health_watchdog.stop()
    if resource_watchdog:
        await resource_watchdog.stop()
    if download_watchdog:
        await download_watchdog.stop()


# --- アプリケーション ---
app = FastAPI(
    title="vLLM Manager",
    description="vLLM サーバー管理 API",
    version="1.0.0",
    lifespan=lifespan,
)

def _cors_origins() -> list[str]:
    raw = os.environ.get("VLLM_MANAGER_CORS_ORIGINS", "").strip()
    if not raw or raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


# --- サーバー管理 API ---

@router.post("/api/auth/login")
async def api_login(req: LoginRequest, request: Request):
    client_ip = request.client.host if request.client else None
    check_login_allowed(req.username, client_ip)
    user = authenticate(req.username, req.password)
    if not user:
        record_login_failure(req.username, client_ip)
        raise HTTPException(status_code=401, detail="Invalid username or password")
    record_login_success(req.username, client_ip)
    token = create_session(user)
    public = public_user(user)
    public["must_change_password"] = user_must_change_password(public["username"])
    if public.get("role") == "admin":
        public = {**public, "security_warnings": collect_security_warnings()}
    return {"token": token, "user": public}


@router.get("/api/auth/me")
async def api_me(user: dict = Depends(get_current_user)):
    result = {**user, "must_change_password": user_must_change_password(user["username"])}
    if user.get("role") == "admin":
        result["security_warnings"] = collect_security_warnings()
    return result


@router.patch("/api/auth/me")
async def api_update_me(req: SelfUserUpdateRequest, user: dict = Depends(get_current_user)):
    fields = req.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        return update_user(
            user["username"],
            password=fields.get("password"),
            litellm_team_id=fields.get("litellm_team_id"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/auth/me/tokens")
async def api_my_tokens(user: dict = Depends(get_current_user)):
    tokens = list_api_keys(username=user["username"])
    return {"tokens": tokens, "count": len(tokens)}


@router.post("/api/auth/me/tokens")
async def api_my_create_token(req: SelfTokenCreateRequest, user: dict = Depends(get_current_user)):
    try:
        public_record, raw_key = create_api_key(
            user["username"],
            req.name,
            expires_in_days=req.expires_in_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await event_bus.publish(
        "api_token_created",
        public_record,
        message=f"API token created: {req.name}",
        actor=user["username"],
    )
    return {**public_record, "token": raw_key}


@router.delete("/api/auth/me/tokens/{token_id}")
async def api_my_revoke_token(token_id: str, user: dict = Depends(get_current_user)):
    if not revoke_api_key(token_id, username=user["username"]):
        raise HTTPException(status_code=404, detail="Token not found")
    await event_bus.publish(
        "api_token_revoked",
        {"id": token_id},
        message="API token revoked",
        actor=user["username"],
    )
    return {"success": True, "id": token_id}


@router.get("/api/auth/me/api-keys")
async def api_my_litellm_keys(user: dict = Depends(get_current_user)):
    user_id = user.get("litellm_user_id") or user["username"]
    team_id = user.get("litellm_team_id")
    detailed = await _litellm_list_keys_detailed()
    keys = []
    for entry in detailed:
        info = entry.get("info", {})
        if not isinstance(info, dict):
            continue
        if info.get("user_id") == user_id or (team_id and info.get("team_id") == team_id):
            keys.append(entry)
    return {"keys": keys, "count": len(keys)}


@router.post("/api/auth/me/api-keys")
async def api_my_litellm_create_key(req: SelfApiKeyRequest, user: dict = Depends(get_current_user)):
    models = _sanitize_models(req.models)
    payload: dict[str, Any] = {
        "user_id": user.get("litellm_user_id") or user["username"],
        "team_id": user.get("litellm_team_id") or None,
        "models": models,
    }
    if req.key_alias and req.key_alias.strip():
        payload["key_alias"] = req.key_alias.strip()
    if req.max_budget is not None:
        payload["max_budget"] = req.max_budget
    if req.budget_duration:
        payload["budget_duration"] = req.budget_duration
    if req.rpm_limit is not None:
        payload["rpm_limit"] = req.rpm_limit
    if req.tpm_limit is not None:
        payload["tpm_limit"] = req.tpm_limit
    data = await litellm_request("POST", "/key/generate", payload)
    await event_bus.publish("litellm_key_updated", data, message="self key generated", actor=user["username"])
    return data


@router.post("/api/auth/me/api-keys/ensure")
async def api_my_litellm_ensure_key(req: SelfApiKeyEnsureRequest, user: dict = Depends(get_current_user)):
    """Get-or-create LiteLLM sk- for CLI/agents (same alias + models → reuse)."""
    models = _sanitize_models(req.models)
    try:
        result = await ensure_cli_inference_key(
            user,
            models=models,
            key_alias=req.key_alias,
            force=req.force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        detail = (exc.response.text or str(exc))[:500]
        raise HTTPException(status_code=502, detail=f"LiteLLM key ensure failed: {detail}") from exc
    await event_bus.publish(
        "litellm_key_updated",
        {"key_alias": result.get("key_alias"), "reused": result.get("reused"), "models": result.get("models")},
        message="cli key ensured",
        actor=user["username"],
    )
    return result


@router.get("/api/users/{username}/api-keys")
async def api_user_litellm_keys(username: str, admin: dict = Depends(require_admin)):
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    u = public_user(users[username])
    user_id = u.get("litellm_user_id") or u["username"]
    team_id = u.get("litellm_team_id")

    detailed = await _litellm_list_keys_detailed()
    keys = []
    for entry in detailed:
        info = entry.get("info", {})
        if not isinstance(info, dict):
            continue
        if info.get("user_id") == user_id or (team_id and info.get("team_id") == team_id):
            keys.append(entry)
    return {"keys": keys, "count": len(keys), "user_id": user_id, "team_id": team_id}


@router.post("/api/users/{username}/api-keys")
async def api_user_litellm_create_key(username: str, req: AdminUserApiKeyRequest, admin: dict = Depends(require_admin)):
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    u = public_user(users[username])
    models = _sanitize_models(req.models)
    payload: dict[str, Any] = {
        "user_id": u.get("litellm_user_id") or u["username"],
        "team_id": u.get("litellm_team_id") or None,
        "models": models,
    }
    if req.key_alias and req.key_alias.strip():
        payload["key_alias"] = req.key_alias.strip()
    if req.max_budget is not None:
        payload["max_budget"] = req.max_budget
    if req.budget_duration:
        payload["budget_duration"] = req.budget_duration
    if req.rpm_limit is not None:
        payload["rpm_limit"] = req.rpm_limit
    if req.tpm_limit is not None:
        payload["tpm_limit"] = req.tpm_limit
    data = await litellm_request("POST", "/key/generate", payload)
    await event_bus.publish("litellm_key_updated", data, message="user key generated", actor=admin["username"])
    return data


@router.get("/api/users/{username}/tokens")
async def api_admin_user_tokens(username: str, _: dict = Depends(require_admin)):
    """管理者による他ユーザーの永続APIトークン（PAT）一覧確認。"""
    users = load_users()
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    tokens = list_api_keys(username=username)
    return {"tokens": tokens, "count": len(tokens)}


@router.delete("/api/users/{username}/tokens/{token_id}")
async def api_admin_revoke_user_token(username: str, token_id: str, admin: dict = Depends(require_admin)):
    """管理者による他ユーザーの永続APIトークン（PAT）の強制失効。"""
    if not revoke_api_key(token_id, username=username):
        raise HTTPException(status_code=404, detail="Token not found")
    await event_bus.publish(
        "api_token_revoked",
        {"id": token_id, "username": username},
        message=f"管理者が {username} のAPIトークンを失効させました",
        actor=admin["username"],
    )
    return {"success": True, "id": token_id, "username": username}


def _litellm_inference_url() -> str:
    return os.environ.get("LITELLM_INTERNAL_URL", "http://litellm:4000").rstrip("/")


def _chat_eligible_servers(servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """チャット UI に載せる vLLM プロセス（embedding 除外・ポート付きのみ）。"""
    eligible: list[dict[str, Any]] = []
    for server in servers:
        port = server.get("port")
        if port is None or _is_backend_self_port(port):
            continue
        if str(server.get("task_type") or "chat") != "chat":
            continue
        eligible.append(server)
    return eligible


async def _chat_ui_model_ids(user: dict[str, Any]) -> list[str]:
    username = str(user.get("username") or "")
    chat_key = await get_or_create_chat_key(user)
    if chat_key_allows_wildcard(username):
        allowed = ["*"]
    else:
        allowed = await _allowed_models_from_litellm_key(chat_key)
        if allowed is None:
            raise HTTPException(status_code=401, detail="Unable to verify chat API key")

    servers = list_running_servers()
    active_servers = _chat_eligible_servers(servers)
    status = get_status()
    if not active_servers and (not status.get("running") or not status.get("vllm_port")):
        return []
    if not active_servers and status.get("running") and status.get("vllm_port"):
        status_port = int(status["vllm_port"])
        if not _is_backend_self_port(status_port):
            active_servers = [
                {
                    "port": status_port,
                    "model": status.get("model"),
                    "task_type": "chat",
                }
            ]

    try:
        all_models = await _collect_active_vllm_models(active_servers=active_servers, status=status)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise HTTPException(status_code=503, detail="vLLM upstream is unavailable")

    normalized_allowed = {_normalize_model_id(m) for m in allowed if _normalize_model_id(m)}
    model_ids: list[str] = []
    if "*" in normalized_allowed:
        model_ids = [str(m.get("id")) for m in all_models if m.get("id")]
    else:
        for entry in all_models:
            model_id = str(entry.get("id", ""))
            if model_id and _model_allowed_for_key(model_id, normalized_allowed):
                model_ids.append(model_id)

    if not model_ids:
        for server in active_servers:
            model_id = str(server.get("model") or "").strip()
            if model_id and model_id not in model_ids:
                model_ids.append(model_id)

    if not model_ids and (all_models or active_servers):
        return ["vllm-local"]
    return model_ids


async def _proxy_chat_to_litellm(
    *,
    chat_key: str,
    payload: dict[str, Any],
    stream: bool,
) -> Response | StreamingResponse:
    target = f"{_litellm_inference_url()}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {chat_key}",
        "Content-Type": "application/json",
    }
    body = json.dumps(payload).encode("utf-8")
    timeout = _proxy_upstream_timeout()

    if stream:
        client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        try:
            req = client.build_request("POST", target, content=body, headers=headers)
            upstream = await client.send(req, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            with suppress(Exception):
                await client.aclose()
            raise HTTPException(status_code=503, detail="LiteLLM upstream is unavailable")

        if upstream.status_code in {401, 403}:
            error_body = await upstream.aread()
            with suppress(Exception):
                await upstream.aclose()
            with suppress(Exception):
                await client.aclose()
            return Response(
                content=error_body,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
            )

        response_headers = {}
        for key, value in upstream.headers.items():
            lk = key.lower()
            if lk in {"content-length", "transfer-encoding", "connection", "content-encoding"}:
                continue
            response_headers[key] = value

        async def _iter_upstream():
            try:
                async for chunk in upstream.aiter_raw():
                    if chunk:
                        yield chunk
            finally:
                with suppress(Exception):
                    await upstream.aclose()
                with suppress(Exception):
                    await client.aclose()

        return StreamingResponse(
            _iter_upstream(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            upstream = await client.post(target, content=body, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise HTTPException(status_code=503, detail="LiteLLM upstream is unavailable")

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@router.get("/api/chat/models")
async def api_chat_models(user: dict = Depends(get_current_user)):
    try:
        models = await _chat_ui_model_ids(user)
    except httpx.HTTPStatusError as exc:
        detail = "チャット用 API キーの準備に失敗しました。しばらくして再試行してください。"
        try:
            body = exc.response.json()
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                msg = body["error"].get("message")
                if isinstance(msg, str) and msg.strip():
                    detail = msg.strip()
        except Exception:
            pass
        raise HTTPException(status_code=503, detail=detail) from exc
    return {"models": models, "count": len(models)}


@router.post("/api/chat/completions")
async def api_chat_completions(req: ChatUiCompletionRequest, user: dict = Depends(get_current_user)):
    payload: dict[str, Any] = {
        "model": req.model.strip(),
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "stream": req.stream,
    }
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    chat_key = await get_or_create_chat_key(user)
    result = await _proxy_chat_to_litellm(chat_key=chat_key, payload=payload, stream=req.stream)

    if result.status_code in {401, 403}:
        chat_key = await regenerate_chat_key(user)
        result = await _proxy_chat_to_litellm(chat_key=chat_key, payload=payload, stream=req.stream)

    if isinstance(result, StreamingResponse):
        return result

    if result.status_code >= 400:
        detail = result.body.decode("utf-8", errors="replace") if result.body else result.status_code
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and parsed.get("error"):
                err = parsed["error"]
                detail = err.get("message") if isinstance(err, dict) else str(err)
        except Exception:
            pass
        raise HTTPException(status_code=result.status_code, detail=detail or "Chat completion failed")

    return Response(
        content=result.body,
        status_code=result.status_code,
        media_type=result.media_type or "application/json",
    )


@router.get("/api/users")
async def api_users(_: dict = Depends(require_admin)):
    return [public_user(user) for user in load_users().values()]


@router.post("/api/users")
async def api_create_user(req: UserCreateRequest, admin: dict = Depends(require_admin)):
    try:
        user = upsert_user(
            req.username,
            req.password,
            role=req.role,
            litellm_user_id=req.litellm_user_id,
            litellm_team_id=req.litellm_team_id,
        )
        await event_bus.publish("user_updated", user, message="user saved", actor=admin["username"])
        return user
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/api/users/{username}")
async def api_update_user(username: str, req: UserUpdateRequest, admin: dict = Depends(require_admin)):
    try:
        fields = req.model_dump(exclude_unset=True)
        user = update_user(
            username,
            password=fields.get("password"),
            role=fields.get("role"),
            litellm_user_id=fields.get("litellm_user_id"),
            litellm_team_id=fields.get("litellm_team_id"),
            disabled=fields.get("disabled"),
        )
        await event_bus.publish("user_updated", user, message="user updated", actor=admin["username"])
        return user
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/api/status", response_model=ServerStatusResponse)
async def api_status():
    """サーバーの現在状態を取得"""
    return get_status()


@router.get("/api/health/check")
async def api_health_check(_: dict = Depends(get_current_user)):
    """HTTP のみの手動ヘルスチェック（推論キューには載せない）。"""
    return await check_all_services()


@router.post("/api/start", response_model=ApiResponse)
async def api_start(req: ServerStartRequest, admin: dict = Depends(require_admin)):
    """vLLM サーバーを起動"""
    catalog = {model["id"]: model for model in load_model_catalog()}
    if req.model_id not in catalog:
        raise HTTPException(status_code=400, detail="Model must be registered before it can be started")
    catalog_entry = catalog[req.model_id]
    resolved = _resolve_start_request(req, catalog_entry)
    task_type = resolved["task_type"]
    trust_remote_code = resolved["trust_remote_code"]
    await event_bus.publish(
        "server_job",
        {"status": "starting", "model_id": req.model_id, "task_type": task_type},
        actor=admin["username"],
    )
    result = start_server(
        model_id=req.model_id,
        context_length=resolved["context_length"],
        max_num_seqs=req.max_num_seqs,
        default_max_tokens=req.default_max_tokens,
        default_temperature=req.default_temperature,
        default_top_p=req.default_top_p,
        default_frequency_penalty=req.default_frequency_penalty,
        default_presence_penalty=req.default_presence_penalty,
        gpu_memory_mode=req.gpu_memory_mode,
        gpu_memory_utilization=req.gpu_memory_utilization,
        tensor_parallel_size=req.tensor_parallel_size,
        gpu_devices=req.gpu_devices,
        speculative_config=req.speculative_config,
        download_model=req.download_model,
        enable_auto_tool_choice=req.enable_auto_tool_choice,
        tool_call_parser=req.tool_call_parser,
        force_stream=req.force_stream,
        limit_mm_per_prompt=req.limit_mm_per_prompt,
        mm_encoder_tp_mode=req.mm_encoder_tp_mode or None,
        mm_processor_cache_type=req.mm_processor_cache_type or None,
        task_type=task_type,
        trust_remote_code=trust_remote_code,
        enable_lora=req.enable_lora,
        max_lora_rank=req.max_lora_rank,
        instance_id=req.instance_id,
        instance_name=req.instance_name,
        create_new_instance=resolved["create_new_instance"],
    )
    await event_bus.publish(
        "server_job",
        {"status": "completed" if result["success"] else "failed", "result": result},
        message=result["message"],
        actor=admin["username"],
    )
    return ApiResponse(success=result["success"], message=result["message"], steps=result.get("steps", []))


@router.post("/api/stop")
async def api_stop(admin: dict = Depends(require_admin)):
    """vLLM サーバーを停止"""
    result = stop_server()
    await event_bus.publish("server_job", {"status": "stopped", "result": result}, message=result["message"], actor=admin["username"])
    return ApiResponse(success=result["success"], message=result["message"])


@router.get("/api/servers")
async def api_servers(_: dict = Depends(require_admin)):
    """起動中の vLLM サーバー一覧を取得"""
    return list_running_servers()


@router.get("/api/instances")
async def api_instances(_: dict = Depends(require_admin)):
    """管理対象 vLLM インスタンス一覧を取得"""
    return list_instances()


@router.post("/api/instances/stop")
async def api_stop_instance(req: StopInstanceRequest, admin: dict = Depends(require_admin)):
    """指定 instance_id の vLLM サーバーを停止"""
    result = stop_instance(req.instance_id)
    await event_bus.publish(
        "server_job",
        {"status": "stopped", "result": result, "instance_id": req.instance_id},
        message=result["message"],
        actor=admin["username"],
    )
    return result


@router.post("/api/instances/{instance_id}/smoke-test")
async def api_instance_smoke_test(instance_id: str, admin: dict = Depends(require_admin)):
    """稼働中インスタンスへ最小リクエストを送り、実際に応答を返せるか検証する。"""
    result = await run_smoke_test(instance_id)
    await event_bus.publish(
        "instance_smoke_test",
        result,
        message=(
            f"疎通テスト成功 (instance={instance_id}, {result.get('latency_ms')}ms)"
            if result.get("success")
            else f"疎通テスト失敗 (instance={instance_id}): {result.get('error')}"
        ),
        actor=admin["username"],
    )
    return result


@router.post("/api/servers/stop")
async def api_stop_server_by_pid(req: StopServerByPidRequest, admin: dict = Depends(require_admin)):
    """指定 PID の vLLM サーバーを停止"""
    result = stop_server_by_pid(req.pid)
    await event_bus.publish("server_job", {"status": "stopped", "result": result}, message=result["message"], actor=admin["username"])
    return result


@router.post("/api/restart")
async def api_restart(admin: dict = Depends(require_admin)):
    """vLLM サーバーを再起動"""
    await event_bus.publish("server_job", {"status": "restarting"}, actor=admin["username"])
    result = restart_server()
    await event_bus.publish(
        "server_job",
        {"status": "completed" if result["success"] else "failed", "result": result},
        message=result["message"],
        actor=admin["username"],
    )
    return ApiResponse(success=result["success"], message=result["message"], steps=result.get("steps", []))


@router.get("/api/config")
async def api_config():
    """現在設定を取得"""
    return load_config()


@router.get("/api/system-metrics")
async def api_system_metrics():
    """ホストのシステム使用率を取得"""
    return get_system_metrics()


@router.get("/api/storage")
async def api_storage_overview(_: dict = Depends(require_admin)):
    """ドライブごとの使用量サマリ（即答）"""
    return storage_info.get_overview()


@router.get("/api/storage/usage")
async def api_storage_usage(refresh: bool = False, _: dict = Depends(require_admin)):
    """用途別の内訳（HF キャッシュ: モデル別 / Ollama: モデル別 / 学習ジョブ: ジョブ別）"""
    import anyio

    return await anyio.to_thread.run_sync(lambda: storage_info.get_usage_report(refresh=refresh))


@router.get("/api/storage/breakdown")
async def api_storage_breakdown(
    path: str = Query(min_length=1, description="ホスト視点のパス（例: /home）"),
    refresh: bool = False,
    timeout_sec: float = Query(default=300.0, ge=10.0, le=1800.0),
    top: int = Query(default=40, ge=1, le=200),
    _: dict = Depends(require_admin),
):
    """ディレクトリ直下の容量内訳（du。大きなツリーは分単位、結果はキャッシュ）"""
    import anyio

    result = await anyio.to_thread.run_sync(
        lambda: storage_info.get_breakdown(path, refresh=refresh, timeout_sec=timeout_sec, top=top)
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.get("/api/log")
async def api_log(tail: int = 100, instance_id: Optional[str] = None, _: dict = Depends(require_admin)):
    """vLLM サーバーのログを取得"""
    return {"log": get_log_lines(tail=tail, instance_id=instance_id)}


@router.get("/api/admin/litellm-proxy-requests/{track_id}")
async def api_litellm_proxy_request_detail(track_id: str, _: dict = Depends(require_admin)):
    """処理中 LiteLLM リクエストの詳細（プロンプト全文）"""
    detail = get_active_detail(track_id)
    if detail is None:
        hist = get_history_record(track_id)
        if hist is None:
            raise HTTPException(status_code=404, detail="Request not found")
        return hist
    return detail


# --- 学習ジョブ API（LoRA SFT / DPO / GRPO） ---


class TrainingJobRequest(BaseModel):
    method: str = Field(pattern="^(sft|dpo|grpo)$")
    base_model: str = Field(min_length=1)
    dataset: str = Field(min_length=1, description="アップロード済み *.jsonl 名 or HF dataset id")
    gpu_devices: str = Field(min_length=1, description="'1' や '0,1' の形式で明示指定")
    job_name: Optional[str] = None
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    quantization: str = Field(default="4bit", pattern="^(4bit|8bit|none)$")
    reward: Optional[dict[str, Any]] = None
    min_free_gb: float = Field(default=16.0, ge=0.0, le=96.0)


class DeployAdapterRequest(BaseModel):
    port: int = Field(ge=1, le=65535, description="LoRA 有効で起動済みの vLLM ポート")
    lora_name: str = Field(min_length=1, max_length=128)


@router.post("/api/training/datasets")
async def api_training_upload_dataset(file: UploadFile = File(...), _: dict = Depends(require_admin)):
    """学習データセット（JSONL）をアップロード"""
    content = await file.read()
    result = training_manager.save_dataset(file.filename or "", content)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.get("/api/training/datasets")
async def api_training_list_datasets(_: dict = Depends(require_admin)):
    return training_manager.list_datasets()


@router.post("/api/training/jobs")
async def api_training_submit(req: TrainingJobRequest, admin: dict = Depends(require_admin)):
    """学習ジョブを投入（同時実行は 1 ジョブまで）"""
    result = training_manager.submit_job(
        method=req.method,
        base_model=req.base_model,
        dataset=req.dataset,
        gpu_devices=req.gpu_devices,
        job_name=req.job_name,
        hyperparams=req.hyperparams,
        quantization=req.quantization,
        reward=req.reward,
        min_free_gb=req.min_free_gb,
    )
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=result.get("message"))
    await event_bus.publish(
        "training_job",
        {"status": "submitted", "job_id": result["job_id"], "method": req.method},
        actor=admin["username"],
    )
    return result


@router.get("/api/training/jobs")
async def api_training_jobs(_: dict = Depends(require_admin)):
    return training_manager.list_jobs()


@router.get("/api/training/jobs/{job_id}")
async def api_training_job_detail(job_id: str, log_tail: int = 0, _: dict = Depends(require_admin)):
    detail = training_manager.get_job(job_id, log_tail=log_tail)
    if detail is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return detail


@router.get("/api/training/jobs/{job_id}/log")
async def api_training_job_log(job_id: str, tail: int = 200, _: dict = Depends(require_admin)):
    return {"log": training_manager.get_job_log(job_id, tail=tail)}


@router.post("/api/training/jobs/{job_id}/cancel")
async def api_training_cancel(job_id: str, admin: dict = Depends(require_admin)):
    result = training_manager.cancel_job(job_id)
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=result.get("message"))
    await event_bus.publish("training_job", {"status": "cancelled", "job_id": job_id}, actor=admin["username"])
    return result


@router.post("/api/training/jobs/{job_id}/deploy")
async def api_training_deploy(job_id: str, req: DeployAdapterRequest, admin: dict = Depends(require_admin)):
    """学習済み LoRA アダプタを稼働中 vLLM へホットロード。

    対象 vLLM は enable_lora=true で起動されている必要がある
    （/api/start の enable_lora パラメータ）。
    """
    detail = training_manager.get_job(job_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if detail["status"].get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"ジョブが完了していません（status={detail['status'].get('status')}）")
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), trust_env=False) as client:
        upstream = await client.post(
            f"http://127.0.0.1:{req.port}/v1/load_lora_adapter",
            json={"lora_name": req.lora_name, "lora_path": detail["adapter_path"]},
        )
    if upstream.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"vLLM がアダプタロードを拒否しました（HTTP {upstream.status_code}）: {upstream.text[:500]}",
        )
    await event_bus.publish(
        "training_job",
        {"status": "deployed", "job_id": job_id, "lora_name": req.lora_name, "port": req.port},
        actor=admin["username"],
    )
    return {"success": True, "lora_name": req.lora_name, "message": f"アダプタを port {req.port} にロードしました"}


@router.get("/api/admin/request-history")
async def api_request_history(
    limit: int = 50,
    offset: int = 0,
    _: dict = Depends(require_admin),
):
    """完了済み LiteLLM リクエスト履歴"""
    lim = max(1, min(500, limit))
    off = max(0, offset)
    records, total = list_history_records(limit=lim, offset=off)
    return {"requests": records, "total": total, "limit": lim, "offset": off}


@router.get("/api/admin/request-history/{record_id}")
async def api_request_history_detail(record_id: str, _: dict = Depends(require_admin)):
    """完了済みリクエスト 1 件の詳細"""
    record = get_history_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return record


@router.get("/api/models")
async def api_models():
    """利用可能なモデルリストを取得"""
    return load_model_catalog()


@router.post("/api/models")
async def api_register_model(req: ModelRegisterRequest, admin: dict = Depends(require_admin)):
    model = save_model(req.model_dump())
    await event_bus.publish("model_registered", model, message="model registered", actor=admin["username"])
    return model


@router.delete("/api/models/{model_id:path}/cache")
async def api_delete_model_cache(model_id: str, admin: dict = Depends(require_admin)):
    result = delete_model_cache(model_id)
    await event_bus.publish("model_download", result, message="model cache removed", actor=admin["username"])
    return result


@router.delete("/api/models/{model_id:path}")
async def api_delete_model(model_id: str, admin: dict = Depends(require_admin)):
    result = delete_model(model_id)
    await event_bus.publish("model_registered", result, message="model removed", actor=admin["username"])
    return result


@router.get("/api/model-downloads")
async def api_model_downloads(_: dict = Depends(require_admin)):
    reconcile_orphan_download_jobs(actor="system")
    return load_jobs()


@router.post("/api/model-downloads")
async def api_start_model_download(req: DownloadRequest, admin: dict = Depends(require_admin)):
    try:
        return await start_download_job(req.model_id, actor=admin["username"], force=req.force)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/model-downloads/cancel")
async def api_cancel_model_download(req: DownloadRequest, admin: dict = Depends(require_admin)):
    return await cancel_active_download_jobs(req.model_id, actor=admin["username"])


@router.post("/api/model-downloads/resume")
async def api_resume_model_download(req: DownloadRequest, admin: dict = Depends(require_admin)):
    """停止・停滞したジョブをキャンセルし、キャッシュ済みバイトから続きを再開する。"""
    try:
        return await resume_download_job(req.model_id, actor=admin["username"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/context-presets")
async def api_context_presets():
    """コンテキスト長のプリセットを取得"""
    return get_context_presets()


# --- WebSocket メトリクス ---

@router.websocket("/ws/metrics")
async def websocket_metrics(ws: WebSocket, token: str | None = Query(default=None)):
    """イベント WebSocket エンドポイント（互換性のため /ws/metrics を維持）"""
    if not token:
        await ws.close(code=1008, reason="Authentication required")
        return
    try:
        user = get_user_from_token(token)
    except HTTPException:
        await ws.close(code=1008, reason="Invalid or expired session")
        return

    await ws.accept()
    event_bus.register(ws, user)
    is_admin = user.get("role") == "admin"

    try:
        # 接続時に直近の履歴を送信
        await ws.send_json(
            {
                "type": "event_history",
                "data": event_bus.get_history(count=50, user=user),
            }
        )
        history = metrics_scraper.get_history(count=10) if metrics_scraper else []
        if history:
            await ws.send_json({"type": "history", "data": history})
        if is_admin:
            await ws.send_json(
                {"type": "litellm_proxy_snapshot", "data": {"requests": litellm_proxy_snapshot()}}
            )

        while True:
            # クライアントからのメッセージを待機（ping 用）
            try:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_json({"type": "pong"})
            except Exception:
                break
    except WebSocketDisconnect:
        event_bus.unregister(ws)
    except Exception:
        event_bus.unregister(ws)


@router.websocket("/ws/events")
async def websocket_events(ws: WebSocket):
    await websocket_metrics(ws)


# --- LiteLLM 関連 API ---

@router.get("/api/litellm/status")
async def api_litellm_status():
    """LiteLLM の状態を確認"""
    try:
        return await litellm_status()
    except Exception:
        return {"healthy": False}


@router.get("/api/litellm/keys")
async def api_litellm_keys(_: dict = Depends(require_admin)):
    return await litellm_request("GET", "/key/list")


@router.post("/api/litellm/keys")
async def api_litellm_create_key(req: LiteLLMProxyRequest, admin: dict = Depends(require_admin)):
    data = await litellm_request("POST", "/key/generate", req.payload)
    await event_bus.publish("litellm_key_updated", data, message="key generated", actor=admin["username"])
    return data


@router.post("/api/litellm/keys/delete")
async def api_litellm_delete_key(req: LiteLLMProxyRequest, admin: dict = Depends(require_admin)):
    data = await litellm_request("POST", "/key/delete", req.payload)
    await event_bus.publish("litellm_key_updated", data, message="key deleted", actor=admin["username"])
    return data


@router.get("/api/litellm/users")
async def api_litellm_users(_: dict = Depends(require_admin)):
    return await litellm_request("GET", "/user/list")


@router.post("/api/litellm/users")
async def api_litellm_create_user(req: LiteLLMProxyRequest, admin: dict = Depends(require_admin)):
    data = await litellm_request("POST", "/user/new", req.payload)
    await event_bus.publish("litellm_user_updated", data, message="LiteLLM user saved", actor=admin["username"])
    return data


@router.get("/api/litellm/teams")
async def api_litellm_teams(_: dict = Depends(require_admin)):
    return await litellm_request("GET", "/team/list")


@router.post("/api/litellm/teams")
async def api_litellm_create_team(req: LiteLLMProxyRequest, admin: dict = Depends(require_admin)):
    data = await litellm_request("POST", "/team/new", req.payload)
    await event_bus.publish("litellm_team_updated", data, message="team saved", actor=admin["username"])
    return data


@router.get("/api/litellm/spend")
async def api_litellm_spend(_: dict = Depends(require_admin)):
    return await litellm_request("GET", "/spend/logs")


@router.get("/api/audit-log")
async def api_audit_log(limit: int = 100, _: dict = Depends(require_admin)):
    safe_limit = max(1, min(500, int(limit)))
    return {"entries": read_audit_log(safe_limit), "count": safe_limit}


app.include_router(router)


# --- OpenAI互換プロキシ ---
# Docker内で vLLM が 8006 など動的ポート起動しても、
# 外部からは backend(:18000) の /v1/* で常にアクセスできるようにする。
def _normalize_model_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _is_alias_model(value: str) -> bool:
    return value in {"", "vllm-local", "claude-vllm-local", "vllm-any", "auto"}


def _extract_bearer_token(request: Request) -> str:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return ""
    return header[7:].strip()


def _model_allowed_for_key(model_id: str, allowed_models: set[str]) -> bool:
    normalized = _normalize_model_id(model_id)
    if not normalized:
        return False
    if normalized in allowed_models:
        return True

    normalized_leaf = normalized.split("/")[-1]
    if normalized_leaf in allowed_models:
        return True

    for allowed in allowed_models:
        allowed_leaf = allowed.split("/")[-1]
        if allowed_leaf == normalized_leaf:
            return True
    return False


async def _allowed_models_from_litellm_key(api_key: str) -> Optional[list[str]]:
    if not api_key:
        return None
    try:
        payload = await litellm_request("GET", f"/key/info?key={quote(api_key)}")
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    info = payload.get("info")
    holder = info if isinstance(info, dict) else payload
    models = holder.get("models") if isinstance(holder, dict) else None
    if not isinstance(models, list):
        return []
    return [m.strip() for m in models if isinstance(m, str) and m.strip()]


async def _collect_active_vllm_models(
    *,
    active_servers: list[dict[str, Any]],
    status: dict[str, Any],
) -> list[dict[str, Any]]:
    timeout = httpx.Timeout(30.0, connect=5.0)
    all_models: list[dict[str, Any]] = []
    seen: set[str] = set()

    # /v1/models は自己再帰ポートを除外した上で、検出された vLLM を集約する。
    ports = sorted({int(s["port"]) for s in active_servers if s.get("port") is not None})
    if not ports and status.get("vllm_port"):
        status_port = int(status["vllm_port"])
        ports = [status_port] if not _is_backend_self_port(status_port) else []

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        for port in ports:
            try:
                upstream = await client.get(f"http://127.0.0.1:{port}/v1/models")
                upstream.raise_for_status()
                payload = upstream.json()
                for entry in payload.get("data", []) if isinstance(payload, dict) else []:
                    model_id = str(entry.get("id", ""))
                    if not model_id or model_id in seen:
                        continue
                    seen.add(model_id)
                    all_models.append(entry)
            except Exception:
                continue
    return all_models


# 同一モデルが複数ポートで動いている場合のラウンドロビン用カウンタ。
# backend プロセス単体（uvicorn workers=1 前提）のメモリ内状態。
_round_robin_counters: dict[str, int] = {}


def _pick_round_robin(key: str, servers: list[dict[str, Any]]) -> dict[str, Any]:
    if len(servers) == 1:
        return servers[0]
    ordered = sorted(servers, key=lambda s: int(s.get("port") or 0))
    idx = _round_robin_counters.get(key, 0) % len(ordered)
    _round_robin_counters[key] = idx + 1
    return ordered[idx]


def _pick_target_server(
    requested_model: str,
    servers: list[dict[str, Any]],
    *,
    preferred_task_type: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    normalized = _normalize_model_id(requested_model)
    candidates = [s for s in servers if s.get("port")]
    if not candidates:
        return None

    if preferred_task_type:
        typed = [
            s for s in candidates if str(s.get("task_type") or "chat").lower() == preferred_task_type
        ]
        if not typed:
            return None
        candidates = typed

    if not _is_alias_model(normalized):
        exact: list[dict[str, Any]] = []
        suffix: list[dict[str, Any]] = []
        for server in candidates:
            server_model = _normalize_model_id(server.get("model"))
            if not server_model:
                continue
            if normalized == server_model:
                exact.append(server)
            elif "/" in server_model and normalized == server_model.split("/")[-1]:
                suffix.append(server)
            elif "/" in normalized and normalized.split("/")[-1] == server_model.split("/")[-1]:
                suffix.append(server)
        # 同じモデルが複数インスタンス立っている場合はラウンドロビンで分散する。
        # exact 一致を優先し、無ければ suffix 一致（別名/略記マッチ）から選ぶ。
        matched = exact or suffix
        if matched:
            return _pick_round_robin(normalized, matched)

    managed = next((s for s in candidates if s.get("managed_by_app")), None)
    if managed:
        return managed
    return sorted(candidates, key=lambda s: int(s.get("port") or 0))[0]


def _preferred_task_type_for_subpath(subpath: str) -> Optional[str]:
    normalized = subpath.strip("/")
    if normalized == "embeddings":
        return "embedding"
    if normalized in {"score", "rerank", "v1/score", "v1/rerank", "v2/rerank"}:
        return "rerank"
    if normalized in {"chat/completions", "completions", "messages"}:
        return "chat"
    return None


def _is_stream_request_payload(payload: Optional[dict[str, Any]]) -> bool:
    return isinstance(payload, dict) and bool(payload.get("stream"))


def _stringify_content_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _normalize_image_url_block(block: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Responses / Chat の画像ブロックを vLLM 向け image_url 形式へ正規化する。"""
    block_type = str(block.get("type") or "")
    if block_type in {"input_image", "image", "image_url"}:
        image_url = block.get("image_url")
        if isinstance(image_url, dict) and image_url.get("url"):
            return {"type": "image_url", "image_url": {"url": str(image_url["url"])}}
        if isinstance(image_url, str) and image_url.strip():
            return {"type": "image_url", "image_url": {"url": image_url.strip()}}
        url = block.get("url")
        if isinstance(url, str) and url.strip():
            return {"type": "image_url", "image_url": {"url": url.strip()}}
    return None


def _content_blocks_to_chat_content(content: Any) -> Any:
    """Responses API の content blocks を chat/completions 向け content に変換する。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _stringify_content_block(content)

    parts: list[Any] = []
    for block in content:
        if isinstance(block, str):
            if block.strip():
                parts.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            text = _stringify_content_block(block)
            if text:
                parts.append({"type": "text", "text": text})
            continue
        block_type = str(block.get("type") or "")
        image_part = _normalize_image_url_block(block)
        if image_part:
            parts.append(image_part)
            continue
        if block_type in {"input_text", "output_text", "text"}:
            text = _stringify_content_block(block.get("text"))
            if text:
                parts.append({"type": "text", "text": text})
        elif block_type == "tool_result":
            text = _stringify_content_block(block.get("content"))
            if text:
                parts.append({"type": "text", "text": text})
        elif block_type == "tool_use":
            text = _stringify_content_block(block.get("input"))
            if text:
                parts.append({"type": "text", "text": text})
        elif "text" in block:
            text = _stringify_content_block(block.get("text"))
            if text:
                parts.append({"type": "text", "text": text})
        elif "content" in block:
            text = _stringify_content_block(block.get("content"))
            if text:
                parts.append({"type": "text", "text": text})

    if not parts:
        return ""
    if len(parts) == 1 and isinstance(parts[0], dict) and parts[0].get("type") == "text":
        return str(parts[0].get("text") or "")
    return parts


def _content_blocks_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    converted = _content_blocks_to_chat_content(content)
    if isinstance(converted, str):
        return converted
    if isinstance(converted, list):
        parts: list[str] = []
        for block in converted:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, dict) and block.get("type") == "image_url":
                parts.append("[image]")
        return "".join(parts)
    return _stringify_content_block(converted)


def _responses_input_to_chat_messages(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, list):
        return [{"role": "user", "content": _stringify_content_block(input_value)}]

    messages: list[dict[str, Any]] = []
    for item in input_value:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "message":
            role = str(item.get("role") or "user")
            content_value = _content_blocks_to_chat_content(item.get("content"))
            if role == "assistant":
                tool_calls: list[dict[str, Any]] = []
                for block in item.get("content", []) if isinstance(item.get("content"), list) else []:
                    if not isinstance(block, dict):
                        continue
                    if str(block.get("type") or "") == "tool_use":
                        tool_calls.append(
                            {
                                "id": str(block.get("id") or f"call_{len(tool_calls)+1}"),
                                "type": "function",
                                "function": {
                                    "name": str(block.get("name") or "tool"),
                                    "arguments": _stringify_content_block(block.get("input") or {}),
                                },
                            }
                        )
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content_value if isinstance(content_value, str) else _content_blocks_to_text(item.get("content")),
                }
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)
            else:
                messages.append({"role": role, "content": content_value})
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": _stringify_content_block(item.get("output")),
                }
            )
    return messages


def _responses_tools_to_chat_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    tools: list[dict[str, Any]] = []
    for tool in value:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "")
        if tool_type in {"function", "custom"} and isinstance(tool.get("name"), str):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": _stringify_content_block(tool.get("description") or ""),
                        "parameters": tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {"type": "object"},
                    },
                }
            )
        elif tool_type == "function" and isinstance(tool.get("function"), dict):
            tools.append({"type": "function", "function": tool.get("function")})
    return tools


def _responses_to_chat_completions_payload(payload: dict[str, Any]) -> dict[str, Any]:
    chat_payload: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": _responses_input_to_chat_messages(payload.get("input")),
        # responses→chat ブリッジでは非streamに固定し、返却整形を簡潔に保つ
        "stream": False,
    }
    if isinstance(payload.get("tools"), list):
        converted_tools = _responses_tools_to_chat_tools(payload.get("tools"))
        if converted_tools:
            chat_payload["tools"] = converted_tools
    if payload.get("tool_choice") is not None:
        chat_payload["tool_choice"] = payload.get("tool_choice")
    if payload.get("temperature") is not None:
        chat_payload["temperature"] = payload.get("temperature")
    if payload.get("top_p") is not None:
        chat_payload["top_p"] = payload.get("top_p")
    max_out = payload.get("max_output_tokens")
    if max_out is None:
        max_out = payload.get("max_tokens")
    if max_out is not None:
        chat_payload["max_tokens"] = max_out
    return chat_payload


def _chat_completion_to_responses_payload(chat_response: dict[str, Any]) -> dict[str, Any]:
    choice = ((chat_response.get("choices") or [{}])[0]) if isinstance(chat_response, dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    text = _stringify_content_block(message.get("content"))
    tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []

    output_items: list[dict[str, Any]] = []
    if text:
        output_items.append(
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for idx, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        output_items.append(
            {
                "type": "function_call",
                "id": str(tc.get("id") or f"fc_{idx+1}"),
                "call_id": str(tc.get("id") or f"fc_{idx+1}"),
                "name": str(fn.get("name") or "tool"),
                "arguments": _stringify_content_block(fn.get("arguments") or "{}"),
                "status": "completed",
            }
        )

    usage = chat_response.get("usage") if isinstance(chat_response, dict) and isinstance(chat_response.get("usage"), dict) else {}
    return {
        "id": str(chat_response.get("id") or "resp_backend_bridge"),
        "object": "response",
        "created_at": int(chat_response.get("created") or 0),
        "status": "completed",
        "model": chat_response.get("model"),
        "output": output_items,
        "output_text": text,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
    }


@app.api_route("/v1/{subpath:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_openai_compat(subpath: str, request: Request):
    original_subpath = subpath
    body = await request.body()
    method = request.method.upper()
    servers = list_running_servers()
    # 再帰防止: backend 自身の listen port(既定 8000)は上流候補から除外
    active_servers = [s for s in servers if s.get("port") and not _is_backend_self_port(s.get("port"))]
    status = get_status()

    if not active_servers and (not status.get("running") or not status.get("vllm_port")):
        raise HTTPException(status_code=503, detail="vLLM server is not running")

    if method == "GET" and subpath == "models":
        try:
            all_models = await _collect_active_vllm_models(active_servers=active_servers, status=status)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise HTTPException(status_code=503, detail="vLLM upstream is unavailable")

        if header_marks_litellm(request):
            incoming_key = _extract_bearer_token(request)
            allowed = await _allowed_models_from_litellm_key(incoming_key)
            if allowed is None:
                raise HTTPException(status_code=401, detail="Unable to verify LiteLLM key models")
            normalized_allowed = {_normalize_model_id(m) for m in allowed if _normalize_model_id(m)}
            if "*" not in normalized_allowed:
                all_models = [
                    m for m in all_models if _model_allowed_for_key(str(m.get("id", "")), normalized_allowed)
                ]
        return Response(
            content=json.dumps({"object": "list", "data": all_models}).encode("utf-8"),
            media_type="application/json",
            status_code=200,
        )

    request_payload: Optional[dict[str, Any]] = None
    requested_model = ""
    bridge_responses_to_chat = False
    if method in {"POST", "PUT", "PATCH"} and body:
        try:
            parsed = json.loads(body.decode("utf-8"))
            if isinstance(parsed, dict):
                request_payload = parsed
                requested_model = str(parsed.get("model") or "")
                if subpath == "responses":
                    request_payload = _responses_to_chat_completions_payload(parsed)
                    requested_model = str(request_payload.get("model") or requested_model)
                    subpath = "chat/completions"
                    bridge_responses_to_chat = True
                    body = json.dumps(request_payload).encode("utf-8")
        except Exception:
            request_payload = None

    preferred_task = _preferred_task_type_for_subpath(subpath)
    target_server = _pick_target_server(
        requested_model,
        active_servers,
        preferred_task_type=preferred_task,
    )
    if target_server is not None:
        target_port = int(target_server["port"])
        target_model = target_server.get("model")
    elif preferred_task is None and status.get("vllm_port"):
        target_port = int(status["vllm_port"])
        target_model = status.get("model")
    else:
        raise HTTPException(status_code=503, detail="No active vLLM server found")

    if request_payload is not None and subpath in {"chat/completions", "completions"}:
        if target_model and _is_alias_model(_normalize_model_id(requested_model)):
            request_payload["model"] = target_model
        cfg = load_config()
        defaults = {
            "max_tokens": int(cfg.get("default_max_tokens", 512)),
            "temperature": float(cfg.get("default_temperature", 0.7)),
            "top_p": float(cfg.get("default_top_p", 0.95)),
            "frequency_penalty": float(cfg.get("default_frequency_penalty", 0.0)),
            "presence_penalty": float(cfg.get("default_presence_penalty", 0.0)),
        }
        for key, value in defaults.items():
            if request_payload.get(key) is None:
                request_payload[key] = value
        if _should_force_stream(request, cfg, request_payload=request_payload) and not request_payload.get(
            "stream"
        ):
            request_payload["stream"] = True
        body = json.dumps(request_payload).encode("utf-8")
    elif request_payload is not None and subpath in {
        "embeddings",
        "score",
        "rerank",
        "v1/score",
        "v1/rerank",
        "v2/rerank",
    }:
        if target_model and _is_alias_model(_normalize_model_id(requested_model)):
            request_payload["model"] = target_model
            body = json.dumps(request_payload).encode("utf-8")

    query_items = list(request.query_params.multi_items())
    if original_subpath == "messages":
        query_items = [(k, v) for (k, v) in query_items if str(k).lower() != "beta"]
    query_string = urlencode(query_items)
    upstream_path = getattr(request.state, "upstream_path", None) or f"/v1/{subpath}"
    target = f"http://127.0.0.1:{target_port}{upstream_path}"
    if query_string:
        target = f"{target}?{query_string}"
    forward_headers = {}
    for key, value in request.headers.items():
        lk = key.lower()
        if lk in {"host", "content-length"}:
            continue
        forward_headers[key] = value

    timeout = _proxy_upstream_timeout()
    # ローカル転送は環境プロキシを使わない（squid 経由で 127.0.0.1 が失敗するため）
    if (
        method == "POST"
        and subpath in {"chat/completions", "completions"}
        and body
        and header_marks_litellm(request)
        and not bridge_responses_to_chat
        and not is_inference_health_probe(request_payload)
    ):
        try:
            return await proxy_litellm_tracked_v1(
                subpath=subpath,
                target=target,
                body=body,
                forward_headers=forward_headers,
                timeout=timeout,
                request_payload=request_payload,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise HTTPException(status_code=503, detail="vLLM upstream is unavailable") from None

    if (
        method == "POST"
        and subpath in {"chat/completions", "completions"}
        and _is_stream_request_payload(request_payload)
        and not bridge_responses_to_chat
    ):
        client = httpx.AsyncClient(timeout=timeout, trust_env=False)
        try:
            req = client.build_request(
                request.method,
                target,
                content=body if body else None,
                headers=forward_headers,
            )
            upstream = await client.send(req, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            with suppress(Exception):
                await client.aclose()
            raise HTTPException(status_code=503, detail="vLLM upstream is unavailable")
        except httpx.LocalProtocolError:
            with suppress(Exception):
                await client.aclose()
            raise HTTPException(status_code=400, detail="invalid request headers")

        response_headers = {}
        for key, value in upstream.headers.items():
            lk = key.lower()
            if lk in {"content-length", "transfer-encoding", "connection", "content-encoding"}:
                continue
            response_headers[key] = value

        async def _iter_upstream():
            try:
                async for chunk in upstream.aiter_raw():
                    if chunk:
                        yield chunk
            finally:
                with suppress(Exception):
                    await upstream.aclose()
                with suppress(Exception):
                    await client.aclose()

        return StreamingResponse(
            _iter_upstream(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            upstream = await client.request(
                request.method,
                target,
                content=body if body else None,
                headers=forward_headers,
            )
    except (httpx.ConnectError, httpx.ConnectTimeout):
        # vLLM再起動直後などで接続不能な瞬間は 500 ではなく 503 を返す
        raise HTTPException(status_code=503, detail="vLLM upstream is unavailable")

    # hop-by-hop ヘッダ等は返却しない
    response_headers = {}
    for key, value in upstream.headers.items():
        lk = key.lower()
        if lk in {"content-length", "transfer-encoding", "connection", "content-encoding"}:
            continue
        response_headers[key] = value

    if bridge_responses_to_chat and upstream.status_code < 400:
        try:
            chat_payload = upstream.json()
            responses_payload = _chat_completion_to_responses_payload(chat_payload)
            return Response(
                content=json.dumps(responses_payload).encode("utf-8"),
                status_code=200,
                headers=response_headers,
                media_type="application/json",
            )
        except Exception:
            pass

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


@app.api_route("/score", methods=["POST"])
async def proxy_score_root(request: Request):
    request.state.upstream_path = "/score"
    return await proxy_openai_compat("score", request)


@app.api_route("/rerank", methods=["POST"])
async def proxy_rerank_root(request: Request):
    request.state.upstream_path = "/rerank"
    return await proxy_openai_compat("rerank", request)


@app.api_route("/v2/rerank", methods=["POST"])
async def proxy_v2_rerank(request: Request):
    request.state.upstream_path = "/v2/rerank"
    return await proxy_openai_compat("v2/rerank", request)
