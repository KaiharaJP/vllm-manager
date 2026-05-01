"""
vLLM Manager - FastAPI アプリケーション

API エンドポイント + WebSocket メトリクス配信を提供する。
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.server_manager import (
    get_status,
    start_server,
    stop_server,
    restart_server,
    get_log_lines,
    get_available_models,
    get_context_presets,
    load_config,
)
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
    steps: list[str] = []


# --- グローバル状態 ---
metrics_scraper: Optional[MetricsScraper] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # スタートアップ
    global metrics_scraper
    metrics_scraper = MetricsScraper(
        vllm_metrics_url=f"http://vllm:8001/metrics",
        scrape_interval=5.0,
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

@router.get("/api/status", response_model=ServerStatusResponse)
async def api_status():
    """サーバーの現在状態を取得"""
    return get_status()


@router.post("/api/start", response_model=ApiResponse)
async def api_start(req: ServerStartRequest):
    """vLLM サーバーを起動"""
    result = start_server(
        model_id=req.model_id,
        context_length=req.context_length,
        max_num_seqs=req.max_num_seqs,
        gpu_memory_utilization=req.gpu_memory_utilization,
        tensor_parallel_size=req.tensor_parallel_size,
        download_model=req.download_model,
    )
    return ApiResponse(success=result["success"], message=result["message"], steps=result.get("steps", []))


@router.post("/api/stop")
async def api_stop():
    """vLLM サーバーを停止"""
    result = stop_server()
    return ApiResponse(success=result["success"], message=result["message"])


@router.post("/api/restart")
async def api_restart():
    """vLLM サーバーを再起動"""
    result = restart_server()
    return ApiResponse(success=result["success"], message=result["message"], steps=result.get("steps", []))


@router.get("/api/config")
async def api_config():
    """現在設定を取得"""
    return load_config()


@router.get("/api/log")
async def api_log(tail: int = 100):
    """vLLM サーバーのログを取得"""
    return {"log": get_log_lines(tail=tail)}


@router.get("/api/models")
async def api_models():
    """利用可能なモデルリストを取得"""
    return get_available_models()


@router.get("/api/context-presets")
async def api_context_presets():
    """コンテキスト長のプリセットを取得"""
    return get_context_presets()


# --- WebSocket メトリクス ---

@router.websocket("/ws/metrics")
async def websocket_metrics(ws: WebSocket):
    """メトリクス WebSocket エンドポイント"""
    await ws.accept()
    metrics_scraper.register_client(ws)

    try:
        # 接続時に直近の履歴を送信
        history = metrics_scraper.get_history(count=10)
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
        metrics_scraper.unregister_client(ws)
    except Exception:
        metrics_scraper.unregister_client(ws)


# --- LiteLLM 関連 API ---

@router.get("/api/litellm/status")
async def api_litellm_status():
    """LiteLLM の状態を確認"""
    try:
        import httpx
        resp = await httpx.AsyncClient(timeout=3.0).get("http://litellm:4000/health")
        return {"healthy": resp.status_code == 200}
    except Exception:
        return {"healthy": False}


app.include_router(router)
