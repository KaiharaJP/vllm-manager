"""
学習ジョブ管理（LoRA SFT / DPO / GRPO）。

vLLM インスタンスと同じ方式で、学習スクリプト（scripts/train_job.py）を
サブプロセスとして起動し、/app/data/training/ 以下でジョブ状態を管理する。

ジョブディレクトリ構成:
  training/
    datasets/<name>.jsonl          アップロードされた学習データ
    jobs/<job_id>/
      config.json                  ジョブ設定（投入時に確定）
      status.json                  進捗（runner が更新: queued/running/completed/failed/cancelled）
      train.log                    学習ログ
      job.pid                      ランチャー PID
      adapter/                     学習済み LoRA アダプタ出力
"""

import json
import os
import re
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from app.server_manager import DATA_DIR

TRAINING_DIR = DATA_DIR / "training"
JOBS_DIR = TRAINING_DIR / "jobs"
DATASETS_DIR = TRAINING_DIR / "datasets"

VALID_METHODS = {"sft", "dpo", "grpo"}
VALID_QUANT = {"4bit", "8bit", "none"}
_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

# 1 ジョブのみ同時実行を許可（GPU の取り合い防止）。
# 複数 GPU で並行学習したい場合はこの制約を外すのではなく、
# ジョブ側の gpu_devices を分けた上で明示的に緩和を検討する。
MAX_CONCURRENT_JOBS = 1


def _ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        # PermissionError は別ユーザーの生存プロセス
        return False
    except Exception:
        return False


def _job_pid(job_id: str) -> Optional[int]:
    pid_file = _job_dir(job_id) / "job.pid"
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def _effective_status(job_id: str) -> dict[str, Any]:
    """status.json とプロセス生存状態を突き合わせた実効ステータスを返す。"""
    status = _read_json(_job_dir(job_id) / "status.json") or {"status": "unknown"}
    if status.get("status") in {"queued", "running"}:
        pid = _job_pid(job_id)
        if pid is None or not _process_alive(pid):
            # runner が status を書けずに死んだ（OOM kill 等）
            status["status"] = "failed"
            status["error"] = status.get("error") or "学習プロセスが異常終了しました（ログを確認してください）"
    return status


def _gpu_free_memory_gb(gpu_index: int) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return float(result.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def sanitize_dataset_name(name: str) -> Optional[str]:
    base = os.path.basename((name or "").strip())
    if not base.endswith(".jsonl"):
        return None
    if not _DATASET_NAME_RE.match(base):
        return None
    return base


def save_dataset(name: str, content: bytes) -> dict[str, Any]:
    _ensure_dirs()
    safe = sanitize_dataset_name(name)
    if not safe:
        return {"success": False, "message": "データセット名は英数字・._- のみの *.jsonl にしてください"}
    lines = [ln for ln in content.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        return {"success": False, "message": "空のデータセットです"}
    for i, line in enumerate(lines[:50]):
        try:
            json.loads(line)
        except Exception:
            return {"success": False, "message": f"JSONL として不正です（{i + 1} 行目）"}
    (DATASETS_DIR / safe).write_bytes(content)
    return {"success": True, "name": safe, "rows": len(lines)}


def list_datasets() -> list[dict[str, Any]]:
    _ensure_dirs()
    out = []
    for path in sorted(DATASETS_DIR.glob("*.jsonl")):
        stat = path.stat()
        out.append({"name": path.name, "size_bytes": stat.st_size, "updated_at": stat.st_mtime})
    return out


def _running_jobs() -> list[str]:
    _ensure_dirs()
    running = []
    for path in JOBS_DIR.iterdir():
        if not path.is_dir():
            continue
        if _effective_status(path.name).get("status") in {"queued", "running"}:
            running.append(path.name)
    return running


def submit_job(
    *,
    method: str,
    base_model: str,
    dataset: str,
    gpu_devices: str,
    job_name: Optional[str] = None,
    hyperparams: Optional[dict[str, Any]] = None,
    quantization: str = "4bit",
    reward: Optional[dict[str, Any]] = None,
    min_free_gb: float = 16.0,
) -> dict[str, Any]:
    _ensure_dirs()

    method = (method or "").strip().lower()
    if method not in VALID_METHODS:
        return {"success": False, "message": f"method は {sorted(VALID_METHODS)} のいずれかを指定してください"}
    if quantization not in VALID_QUANT:
        return {"success": False, "message": f"quantization は {sorted(VALID_QUANT)} のいずれかを指定してください"}
    if not (base_model or "").strip():
        return {"success": False, "message": "base_model は必須です"}
    if method == "grpo":
        reward_type = str((reward or {}).get("type") or "")
        if reward_type not in {"exact_match", "contains", "remote"}:
            return {
                "success": False,
                "message": "grpo では reward.type（exact_match / contains / remote）の指定が必須です",
            }
        if reward_type == "remote" and not str((reward or {}).get("url") or "").strip():
            return {"success": False, "message": "reward.type=remote では reward.url が必須です"}

    # GPU 指定は明示必須（推論インスタンスとの取り合いを起こさないため "all" は不可）
    devices = (gpu_devices or "").strip()
    if not devices or devices.lower() == "all" or not re.match(r"^\d+(,\d+)*$", devices):
        return {"success": False, "message": "gpu_devices は '1' や '0,1' の形式で明示指定してください"}

    first_gpu = int(devices.split(",")[0])
    free_gb = _gpu_free_memory_gb(first_gpu)
    if free_gb is not None and free_gb < float(min_free_gb):
        return {
            "success": False,
            "message": f"GPU{first_gpu} の空きが {free_gb:.1f}GB で不足しています（要求 {min_free_gb}GB 以上）。"
            "推論インスタンスを止めるか min_free_gb を調整してください",
        }

    # データセット解決: アップロード済みファイル名 or HF dataset id
    dataset = (dataset or "").strip()
    local = DATASETS_DIR / os.path.basename(dataset)
    if local.suffix == ".jsonl":
        if not local.exists():
            return {"success": False, "message": f"データセット {local.name} が見つかりません（先にアップロードしてください）"}
        dataset_path = str(local)
    elif "/" in dataset:
        dataset_path = dataset  # HF dataset id
    else:
        return {"success": False, "message": "dataset にはアップロード済み *.jsonl 名か HuggingFace dataset id を指定してください"}

    running = _running_jobs()
    if len(running) >= MAX_CONCURRENT_JOBS:
        return {"success": False, "message": f"既に実行中の学習ジョブがあります: {running}"}

    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job_dir = _job_dir(job_id)
    (job_dir / "adapter").mkdir(parents=True, exist_ok=True)

    config = {
        "job_id": job_id,
        "job_name": (job_name or "").strip() or job_id,
        "method": method,
        "base_model": base_model.strip(),
        "dataset": dataset_path,
        "quantization": quantization,
        "hyperparams": hyperparams or {},
        "reward": reward or {},
        "gpu_devices": devices,
        "output_dir": str(job_dir / "adapter"),
        "created_at": time.time(),
    }
    (job_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (job_dir / "status.json").write_text(json.dumps({"status": "queued", "updated_at": time.time()}))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env.setdefault("HF_HOME", "/app/hf-cache")

    log = open(job_dir / "train.log", "w")
    proc = subprocess.Popen(
        ["python3", "/app/scripts/train_job.py", "--job-dir", str(job_dir)],
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )
    (job_dir / "job.pid").write_text(str(proc.pid))
    return {"success": True, "job_id": job_id, "pid": proc.pid, "config": config}


def list_jobs() -> list[dict[str, Any]]:
    _ensure_dirs()
    jobs = []
    for path in sorted(JOBS_DIR.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        config = _read_json(path / "config.json") or {}
        status = _effective_status(path.name)
        jobs.append(
            {
                "job_id": path.name,
                "job_name": config.get("job_name"),
                "method": config.get("method"),
                "base_model": config.get("base_model"),
                "gpu_devices": config.get("gpu_devices"),
                "status": status.get("status"),
                "progress": status.get("progress"),
                "created_at": config.get("created_at"),
            }
        )
    return jobs


def get_job(job_id: str, *, log_tail: int = 0) -> Optional[dict[str, Any]]:
    job_dir = _job_dir(job_id)
    if not job_dir.is_dir():
        return None
    detail: dict[str, Any] = {
        "job_id": job_id,
        "config": _read_json(job_dir / "config.json"),
        "status": _effective_status(job_id),
        "adapter_path": str(job_dir / "adapter"),
    }
    if log_tail > 0:
        detail["log"] = get_job_log(job_id, tail=log_tail)
    return detail


def get_job_log(job_id: str, *, tail: int = 200) -> list[str]:
    log_file = _job_dir(job_id) / "train.log"
    if not log_file.exists():
        return []
    lines = log_file.read_text(errors="replace").splitlines()
    return lines[-max(1, min(2000, tail)):]


def cancel_job(job_id: str) -> dict[str, Any]:
    job_dir = _job_dir(job_id)
    if not job_dir.is_dir():
        return {"success": False, "message": "ジョブが見つかりません"}
    status = _effective_status(job_id)
    if status.get("status") not in {"queued", "running"}:
        return {"success": False, "message": f"実行中ではありません（status={status.get('status')}）"}
    pid = _job_pid(job_id)
    if pid is None:
        return {"success": False, "message": "PID が見つかりません"}
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception as e:
        return {"success": False, "message": f"停止に失敗しました: {e}"}
    (job_dir / "status.json").write_text(
        json.dumps({**status, "status": "cancelled", "updated_at": time.time()})
    )
    return {"success": True, "message": f"ジョブ {job_id} を停止しました"}
