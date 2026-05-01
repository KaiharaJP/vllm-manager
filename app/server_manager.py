"""
vLLM Manager - サーバー管理モジュール

vLLM サーバーの起動/停止/ステータス管理を担う。
Docker 環境内で動作することを前提としている。
"""

import subprocess
import time
import os
import signal
import json
from pathlib import Path
from typing import Optional

# --- パス設定 (Docker 内で共有) ---
DATA_DIR = Path("/app/data")
PID_FILE = DATA_DIR / "vllm.pid"
LOG_FILE = DATA_DIR / "vllm.log"
CONFIG_FILE = DATA_DIR / "config.json"

# --- デフォルトモデルリスト ---
DEFAULT_MODELS = [
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "name": "Llama 3.3 70B Instruct", "size": "70B"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B Instruct", "size": "8B"},
    {"id": "Qwen/Qwen2.5-72B-Instruct", "name": "Qwen 2.5 72B Instruct", "size": "72B"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5 7B Instruct", "size": "7B"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral 7B Instruct v0.3", "size": "7B"},
    {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3 Mini 4K Instruct", "size": "3.8B"},
    {"id": "google/gemma-2-27b-it", "name": "Gemma 2 27B IT", "size": "27B"},
    {"id": "google/gemma-2-9b-it", "name": "Gemma 2 9B IT", "size": "9B"},
]

# --- コンテキスト長のプリセット ---
CONTEXT_PRESETS = [
    {"value": 4096, "label": "4K"},
    {"value": 8192, "label": "8K"},
    {"value": 32768, "label": "32K"},
    {"value": 65536, "label": "64K"},
    {"value": 131072, "label": "128K"},
]


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """保存済み設定を読み込む。"""
    ensure_data_dir()
    defaults = {
        "model_id": DEFAULT_MODELS[0]["id"],
        "context_length": 8192,
        "max_num_seqs": 256,
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": 1,
        "vllm_port": 8001,
    }
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text())
            defaults.update(saved)
            return defaults
        except (json.JSONDecodeError, IOError):
            pass
    return defaults


def save_config(config: dict) -> None:
    """設定を保存する。"""
    ensure_data_dir()
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_status() -> dict:
    """vLLM サーバーの状態を確認する。"""
    status = {
        "running": False,
        "healthy": False,
        "pid": None,
        "vllm_port": 8001,
        "model": None,
        "uptime_seconds": 0,
    }

    if not PID_FILE.exists():
        return status

    try:
        pid = int(PID_FILE.read_text().strip())
        status["pid"] = pid
        status["vllm_port"] = load_config().get("vllm_port", 8001)

        if not _process_exists(pid):
            PID_FILE.unlink(missing_ok=True)
            return status

        status["running"] = True

        # ヘルスチェック
        try:
            import httpx
            resp = httpx.get(
                f"http://localhost:{status['vllm_port']}/health",
                timeout=3.0,
            )
            if resp.status_code == 200:
                status["healthy"] = True
                health_data = resp.json()
                status["model"] = health_data.get("model")
                try:
                    import psutil
                    proc = psutil.Process(pid)
                    status["uptime_seconds"] = time.time() - proc.create_time()
                except Exception:
                    pass
        except Exception:
            pass

    except (ValueError, IOError):
        PID_FILE.unlink(missing_ok=True)

    return status


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _build_command(config: dict) -> list:
    """vLLM 起動コマンドを構築する。"""
    return [
        "vllm", "serve", config["model_id"],
        "--host", "0.0.0.0",
        "--port", str(config["vllm_port"]),
        "--max-model-len", str(config["context_length"]),
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--tensor-parallel-size", str(config["tensor_parallel_size"]),
        "--disable-log-requests",
    ]


def start_server(
    model_id: Optional[str] = None,
    context_length: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    tensor_parallel_size: Optional[int] = None,
    vllm_port: Optional[int] = None,
    download_model: bool = True,
) -> dict:
    """vLLM サーバーを起動する。None のパラメータは保存済み設定から読み込む。"""
    result = {"success": False, "message": "", "steps": []}
    config = load_config()

    # 渡されたパラメータで上書き
    if model_id is not None:
        config["model_id"] = model_id
    if context_length is not None:
        config["context_length"] = context_length
    if max_num_seqs is not None:
        config["max_num_seqs"] = max_num_seqs
    if gpu_memory_utilization is not None:
        config["gpu_memory_utilization"] = gpu_memory_utilization
    if tensor_parallel_size is not None:
        config["tensor_parallel_size"] = tensor_parallel_size
    if vllm_port is not None:
        config["vllm_port"] = vllm_port

    # 既存サーバーを停止
    status = get_status()
    if status["running"]:
        result["steps"].append("既存サーバーを停止中...")
        stop_server()
        time.sleep(2)

    save_config(config)

    # モデルのダウンロード確認
    if download_model:
        result["steps"].append(f"モデルを確認中: {config['model_id']}")
        try:
            from huggingface_hub import snapshot_download
            hf_home = os.environ.get("HF_HOME", "/root/.cache/huggingface")
            snapshot_download(
                repo_id=config["model_id"],
                cache_dir=hf_home,
            )
            result["steps"].append(f"モデル準備完了: {config['model_id']}")
        except Exception as e:
            result["message"] = f"モデルダウンロード失敗: {str(e)}"
            return result

    # サーバー起動
    cmd = _build_command(config)
    result["steps"].append(f"vLLM サーバーを起動中 (port {config['vllm_port']})")
    result["steps"].append(f"コマンド: {' '.join(cmd)}")

    try:
        log = open(LOG_FILE, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        PID_FILE.write_text(str(proc.pid))
        result["success"] = True
        result["message"] = f"vLLM サーバー起動完了 (PID: {proc.pid})"
        result["steps"].append(f"PID: {proc.pid}")

        time.sleep(3)
        status = get_status()
        if status["healthy"]:
            result["steps"].append("サーバー正常に動作しています！")
        else:
            result["steps"].append("サーバー起動中、初期化に時間がかかる場合があります...")

    except Exception as e:
        result["message"] = f"サーバー起動失敗: {str(e)}"

    return result


def stop_server() -> dict:
    """vLLM サーバーを停止する。"""
    result = {"success": False, "message": ""}

    if not PID_FILE.exists():
        result["message"] = "実行中のサーバーはありません"
        result["success"] = True
        return result

    try:
        pid = int(PID_FILE.read_text().strip())
        if _process_exists(pid):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(2)
            if _process_exists(pid):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            result["message"] = f"サーバー停止完了 (PID: {pid})"
        else:
            result["message"] = "サーバーは実行されていませんでした"
        result["success"] = True
    except Exception as e:
        result["message"] = f"サーバー停止エラー: {str(e)}"

    PID_FILE.unlink(missing_ok=True)
    return result


def restart_server() -> dict:
    """現在の設定でサーバーを再起動する。"""
    config = load_config()
    stop_result = stop_server()
    time.sleep(2)
    start_result = start_server(download_model=False)
    return {
        "success": start_result["success"],
        "message": f"停止: {stop_result['message']}\n起動: {start_result['message']}",
        "steps": start_result.get("steps", []),
    }


def get_log_lines(tail: int = 100) -> str:
    """vLLM サーバーのログを取得する。"""
    if not LOG_FILE.exists():
        return "ログファイルが見つかりません。"
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        return "".join(lines[-tail:])
    except IOError:
        return "ログファイルを読み込めませんでした。"


def get_available_models() -> list:
    return DEFAULT_MODELS


def get_context_presets() -> list:
    return CONTEXT_PRESETS
