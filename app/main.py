"""
vLLM Manager - FastAPI アプリケーション

API エンドポイント + WebSocket メトリクス配信を提供する。
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.server_manager import (
    get_status,
    start_server,
    stop_server,
    restart_server,
    get_log_lines,
    get_context_presets,
    load_config,
)
from app.auth import authenticate, create_session, get_current_user, load_users, public_user, require_admin, upsert_user
from app.event_bus import event_bus
from app.litellm_client import litellm_request, status as litellm_status
from app.model_manager import load_jobs, load_model_catalog, save_model, start_download_job
from app.metrics_scraper import MetricsScraper


# --- Pydantic モデル ---

class ServerStartRequest(BaseModel):
    model_id: str
    context_length: int = Field(default=8192, ge=1024, le=131072)
    max_num_seqs: int = Field(default=256, ge=1, le=1024)
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    tensor_parallel_size: int = Field(default=1, ge=1, le=8)
    download_model: bool = True


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


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = Field(default="user", pattern="^(admin|user)$")
    litellm_user_id: Optional[str] = None
    litellm_team_id: Optional[str] = None


class ModelRegisterRequest(BaseModel):
    id: str
    name: Optional[str] = None
    size: Optional[str] = None
    revision: Optional[str] = None
    gated: bool = False
    trust_remote_code: bool = False
    recommended_context_length: int = Field(default=8192, ge=1024)
    required_gpu_memory_gb: Optional[float] = None
    allowed_roles: list[str] = Field(default_factory=lambda: ["admin", "user"])


class DownloadRequest(BaseModel):
    model_id: str


class LiteLLMProxyRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


# --- グローバル状態 ---
metrics_scraper: Optional[MetricsScraper] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # スタートアップ
    global metrics_scraper
    metrics_scraper = MetricsScraper(
        vllm_metrics_url=f"http://localhost:{os.environ.get('VLLM_PORT', '8001')}/metrics",
        scrape_interval=5.0,
        event_publisher=event_bus.publish,
    )
    await metrics_scraper.start()

    yield

    # シャットダウン
    if metrics_scraper:
        await metrics_scraper.stop()


# --- アプリケーション ---
app = FastAPI(
    title="vLLM Manager",
    description="vLLM サーバー管理 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


# --- サーバー管理 API ---

@router.post("/api/auth/login")
async def api_login(req: LoginRequest):
    user = authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_session(user)
    return {"token": token, "user": public_user(user)}


@router.get("/api/auth/me")
async def api_me(user: dict = Depends(get_current_user)):
    return user


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

@router.get("/api/status", response_model=ServerStatusResponse)
async def api_status():
    """サーバーの現在状態を取得"""
    return get_status()


@router.post("/api/start", response_model=ApiResponse)
async def api_start(req: ServerStartRequest, admin: dict = Depends(require_admin)):
    """vLLM サーバーを起動"""
    if req.model_id not in {model["id"] for model in load_model_catalog()}:
        raise HTTPException(status_code=400, detail="Model must be registered before it can be started")
    await event_bus.publish("server_job", {"status": "starting", "model_id": req.model_id}, actor=admin["username"])
    result = start_server(
        model_id=req.model_id,
        context_length=req.context_length,
        max_num_seqs=req.max_num_seqs,
        gpu_memory_utilization=req.gpu_memory_utilization,
        tensor_parallel_size=req.tensor_parallel_size,
        download_model=req.download_model,
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


@router.get("/api/log")
async def api_log(tail: int = 100, _: dict = Depends(require_admin)):
    """vLLM サーバーのログを取得"""
    return {"log": get_log_lines(tail=tail)}


@router.get("/api/models")
async def api_models():
    """利用可能なモデルリストを取得"""
    return load_model_catalog()


@router.post("/api/models")
async def api_register_model(req: ModelRegisterRequest, admin: dict = Depends(require_admin)):
    model = save_model(req.model_dump())
    await event_bus.publish("model_registered", model, message="model registered", actor=admin["username"])
    return model


@router.get("/api/model-downloads")
async def api_model_downloads(_: dict = Depends(require_admin)):
    return load_jobs()


@router.post("/api/model-downloads")
async def api_start_model_download(req: DownloadRequest, admin: dict = Depends(require_admin)):
    try:
        return await start_download_job(req.model_id, actor=admin["username"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/context-presets")
async def api_context_presets():
    """コンテキスト長のプリセットを取得"""
    return get_context_presets()


# --- WebSocket メトリクス ---

@router.websocket("/ws/metrics")
async def websocket_metrics(ws: WebSocket):
    """イベント WebSocket エンドポイント（互換性のため /ws/metrics を維持）"""
    await ws.accept()
    event_bus.register(ws)

    try:
        # 接続時に直近の履歴を送信
        await ws.send_json({"type": "event_history", "data": event_bus.get_history(count=50)})
        history = metrics_scraper.get_history(count=10) if metrics_scraper else []
        if history:
            await ws.send_json({"type": "history", "data": history})

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


app.include_router(router)
