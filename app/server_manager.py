"""
vLLM Manager - サーバー管理モジュール

vLLM サーバーの起動/停止/ステータス管理を担う。
Docker 環境内で動作することを前提としている。
"""

import re
import subprocess
import time
import os
import signal
import json
import uuid
from pathlib import Path
from typing import Any, Optional
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

# --- パス設定 (Docker 内で共有) ---
DATA_DIR = Path(os.environ.get("VLLM_MANAGER_DATA_DIR", "/tmp/vllm-manager-data"))
PID_FILE = DATA_DIR / "vllm.pid"
LOG_FILE = DATA_DIR / "vllm.log"
CONFIG_FILE = DATA_DIR / "config.json"
INSTANCES_DIR = DATA_DIR / "instances"
INSTANCES_REGISTRY_FILE = DATA_DIR / "instances.json"
DEFAULT_INSTANCE_ID = "default"
VALID_TASK_TYPES = frozenset({"chat", "embedding", "rerank"})
POOLING_TASK_TYPES = frozenset({"embedding", "rerank"})

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
    {"id": "jinaai/jina-embeddings-v3", "name": "Jina Embeddings v3", "size": "embed", "task_type": "embedding", "recommended_context_length": 8192},
]

# --- コンテキスト長のプリセット ---
CONTEXT_PRESETS = [
    {"value": 4096, "label": "4K"},
    {"value": 8192, "label": "8K"},
    {"value": 32768, "label": "32K"},
    {"value": 65536, "label": "64K"},
    {"value": 131072, "label": "128K"},
    {"value": 262144, "label": "256K"},
]
MAX_NUM_SEQS_LIMIT = 20
GPU_MEMORY_UTILIZATION_SAFE_MAX = 0.85
GPU_MEMORY_MODES = frozenset({"auto", "manual", "minimal"})


def _normalize_gpu_memory_mode(value: Any) -> str:
    normalized = str(value or "auto").lower()
    return normalized if normalized in GPU_MEMORY_MODES else "auto"


def _clamp_default_sampling_params(config: dict[str, Any]) -> None:
    config["default_max_tokens"] = max(1, min(262144, int(config.get("default_max_tokens", 512))))
    config["default_temperature"] = max(0.0, min(2.0, float(config.get("default_temperature", 0.7))))
    config["default_top_p"] = max(0.0, min(1.0, float(config.get("default_top_p", 0.95))))
    config["default_frequency_penalty"] = max(
        -2.0, min(2.0, float(config.get("default_frequency_penalty", 0.0)))
    )
    config["default_presence_penalty"] = max(
        -2.0, min(2.0, float(config.get("default_presence_penalty", 0.0)))
    )


def _normalize_tool_call_parser(value: Any) -> str:
    """tool_call_parser を CLI に渡せる短い文字列へ正規化する。"""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > 128:
        return text[:128]
    return text


_REJECTION_SAMPLE_ALIASES = {
    "strict": "standard",
    "probabilistic": "standard",
    "standard": "standard",
    "synthetic": "synthetic",
}
_SPEC_METHOD_ALIASES = {
    "qwen3_next_mtp": "mtp",
}

_VISION_MODEL_HINTS = (
    "qwen3-vl",
    "qwen2-vl",
    "qwen3.6",
    "qwen3.5",
    "qwen3_5",
    "qwen2.5-vl",
    "llava",
    "phi-3.5-vision",
    "phi-4-multimodal",
    "internvl",
    "minicpm-v",
    "image-text-to-text",
)


def _model_likely_supports_vision(model_id: str) -> bool:
    lowered = (model_id or "").lower()
    return any(hint in lowered for hint in _VISION_MODEL_HINTS)


def _normalize_limit_mm_per_prompt(value: Any) -> Optional[dict[str, int]]:
    if value is None:
        return None
    if isinstance(value, dict):
        out: dict[str, int] = {}
        for key, raw in value.items():
            if not isinstance(key, str):
                continue
            try:
                out[key] = int(raw)
            except (TypeError, ValueError):
                continue
        return out or None
    return None


def _resolve_limit_mm_for_config(config: dict) -> Optional[dict[str, int]]:
    explicit = _normalize_limit_mm_per_prompt(config.get("limit_mm_per_prompt"))
    if explicit is not None:
        return explicit
    if _model_likely_supports_vision(str(config.get("model_id") or "")):
        return {"image": 1}
    return None


def _normalize_speculative_config(value: Any) -> Optional[dict[str, Any]]:
    """speculative_config を保存・起動に使える形へ正規化する。"""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        normalized[key] = item
    method = normalized.get("method")
    if isinstance(method, str):
        normalized["method"] = _SPEC_METHOD_ALIASES.get(method, method)
    rejection = normalized.get("rejection_sample_method")
    if isinstance(rejection, str):
        mapped = _REJECTION_SAMPLE_ALIASES.get(rejection.strip().lower())
        if mapped:
            normalized["rejection_sample_method"] = mapped
        else:
            normalized.pop("rejection_sample_method", None)
    return normalized or None


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_task_type(value: Any) -> str:
    task = str(value or "chat").strip().lower()
    return task if task in VALID_TASK_TYPES else "chat"


def _sanitize_instance_id(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "").strip()).strip("-")
    return text[:64] if text else ""


def _instance_paths(instance_id: str) -> dict[str, Path]:
    if instance_id == DEFAULT_INSTANCE_ID:
        return {
            "dir": DATA_DIR,
            "pid": PID_FILE,
            "log": LOG_FILE,
            "config": CONFIG_FILE,
        }
    base = INSTANCES_DIR / instance_id
    return {
        "dir": base,
        "pid": base / "vllm.pid",
        "log": base / "vllm.log",
        "config": base / "config.json",
    }


def _default_config_template() -> dict[str, Any]:
    return {
        "model_id": DEFAULT_MODELS[0]["id"],
        "context_length": 131072,
        "max_num_seqs": 6,
        "force_stream": True,
        "default_max_tokens": 512,
        "default_temperature": 0.7,
        "default_top_p": 0.95,
        "default_frequency_penalty": 0.0,
        "default_presence_penalty": 0.0,
        "gpu_memory_mode": "auto",
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "vllm_port": 8001,
        "gpu_devices": "all",
        "speculative_config": None,
        "enable_auto_tool_choice": False,
        "tool_call_parser": "",
        "limit_mm_per_prompt": None,
        "mm_encoder_tp_mode": "",
        "mm_processor_cache_type": "",
        "task_type": "chat",
        "instance_id": DEFAULT_INSTANCE_ID,
        "instance_name": "default",
        "trust_remote_code": False,
    }


def _normalize_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(config)
    payload["max_num_seqs"] = max(1, min(MAX_NUM_SEQS_LIMIT, int(payload.get("max_num_seqs", 1))))
    _clamp_default_sampling_params(payload)
    payload["context_length"] = max(1024, min(262144, int(payload.get("context_length", 8192))))
    payload["gpu_memory_mode"] = _normalize_gpu_memory_mode(payload.get("gpu_memory_mode"))
    payload["gpu_memory_utilization"] = max(
        0.1, min(GPU_MEMORY_UTILIZATION_SAFE_MAX, float(payload.get("gpu_memory_utilization", 0.85)))
    )
    payload["speculative_config"] = _normalize_speculative_config(payload.get("speculative_config"))
    payload["enable_auto_tool_choice"] = bool(payload.get("enable_auto_tool_choice", False))
    payload["tool_call_parser"] = _normalize_tool_call_parser(payload.get("tool_call_parser"))
    payload["limit_mm_per_prompt"] = _normalize_limit_mm_per_prompt(payload.get("limit_mm_per_prompt"))
    payload["mm_encoder_tp_mode"] = str(payload.get("mm_encoder_tp_mode") or "").strip()
    payload["mm_processor_cache_type"] = str(payload.get("mm_processor_cache_type") or "").strip()
    payload["task_type"] = _normalize_task_type(payload.get("task_type"))
    payload["trust_remote_code"] = bool(payload.get("trust_remote_code", False))
    return payload


def load_config(instance_id: Optional[str] = None) -> dict:
    """保存済み設定を読み込む。instance_id 未指定時は UI テンプレート（default）。"""
    ensure_data_dir()
    target_id = instance_id or DEFAULT_INSTANCE_ID
    paths = _instance_paths(target_id)
    defaults = _default_config_template()
    defaults["instance_id"] = target_id
    if paths["config"].exists():
        try:
            saved = json.loads(paths["config"].read_text())
            defaults.update(saved)
            normalized = _normalize_config_payload(defaults)
            normalized["instance_id"] = target_id
            return normalized
        except (json.JSONDecodeError, IOError):
            pass
    return _normalize_config_payload(defaults)


def save_config(config: dict, instance_id: Optional[str] = None) -> None:
    """設定を保存する。instance_id 未指定時は UI テンプレート（default）。"""
    ensure_data_dir()
    target_id = instance_id or str(config.get("instance_id") or DEFAULT_INSTANCE_ID)
    paths = _instance_paths(target_id)
    if target_id != DEFAULT_INSTANCE_ID:
        paths["dir"].mkdir(parents=True, exist_ok=True)
    payload = _normalize_config_payload(dict(config))
    payload["instance_id"] = target_id
    paths["config"].write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_instances_registry() -> list[dict[str, Any]]:
    ensure_data_dir()
    if not INSTANCES_REGISTRY_FILE.exists():
        return []
    try:
        data = json.loads(INSTANCES_REGISTRY_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and item.get("instance_id")]


def _save_instances_registry(entries: list[dict[str, Any]]) -> None:
    ensure_data_dir()
    INSTANCES_REGISTRY_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


def _upsert_instance_registry(entry: dict[str, Any]) -> None:
    entries = _load_instances_registry()
    instance_id = str(entry.get("instance_id") or "")
    if not instance_id:
        return
    kept = [item for item in entries if item.get("instance_id") != instance_id]
    kept.append(entry)
    _save_instances_registry(kept)


def _remove_instance_registry(instance_id: str) -> None:
    kept = [item for item in _load_instances_registry() if item.get("instance_id") != instance_id]
    _save_instances_registry(kept)


def _registry_entry_for(instance_id: str) -> Optional[dict[str, Any]]:
    for item in _load_instances_registry():
        if item.get("instance_id") == instance_id:
            return item
    return None


def _resolve_new_instance_id(
    *,
    instance_id: Optional[str],
    instance_name: Optional[str],
    model_id: str,
) -> str:
    if instance_id:
        sanitized = _sanitize_instance_id(instance_id)
        if sanitized:
            return sanitized
    if instance_name:
        sanitized = _sanitize_instance_id(instance_name)
        if sanitized:
            return sanitized
    leaf = _sanitize_instance_id(model_id.split("/")[-1]) or "model"
    return f"inst-{leaf}-{uuid.uuid4().hex[:6]}"


def _choose_instance_id_for_legacy_start() -> str:
    default_status = get_instance_status(DEFAULT_INSTANCE_ID)
    if not default_status.get("running"):
        return DEFAULT_INSTANCE_ID
    return _resolve_new_instance_id(instance_id=None, instance_name=None, model_id="vllm")


def get_instance_status(instance_id: str) -> dict[str, Any]:
    """指定インスタンスの vLLM 状態を確認する。"""
    paths = _instance_paths(instance_id)
    status: dict[str, Any] = {
        "instance_id": instance_id,
        "instance_name": (_registry_entry_for(instance_id) or {}).get("instance_name"),
        "task_type": load_config(instance_id).get("task_type", "chat"),
        "running": False,
        "healthy": False,
        "pid": None,
        "vllm_port": load_config(instance_id).get("vllm_port", 8001),
        "model": load_config(instance_id).get("model_id"),
        "uptime_seconds": 0,
    }
    registry = _registry_entry_for(instance_id)
    if registry:
        status["instance_name"] = registry.get("instance_name") or status["instance_name"]
        status["task_type"] = registry.get("task_type") or status["task_type"]

    if not paths["pid"].exists():
        return status

    try:
        pid = int(paths["pid"].read_text().strip())
        status["pid"] = pid
        cfg = load_config(instance_id)
        status["vllm_port"] = cfg.get("vllm_port", 8001)
        status["model"] = cfg.get("model_id")
        status["task_type"] = cfg.get("task_type", status["task_type"])

        if not _process_exists(pid):
            paths["pid"].unlink(missing_ok=True)
            return status

        status["running"] = True
        try:
            import httpx

            resp = httpx.get(
                f"http://localhost:{status['vllm_port']}/health",
                timeout=3.0,
            )
            if resp.status_code == 200:
                status["healthy"] = True
                try:
                    health_data = resp.json()
                    status["model"] = health_data.get("model") or status["model"]
                except Exception:
                    pass
            try:
                import psutil

                proc = psutil.Process(pid)
                status["uptime_seconds"] = time.time() - proc.create_time()
            except Exception:
                pass
        except Exception:
            pass
    except (ValueError, IOError):
        paths["pid"].unlink(missing_ok=True)

    return status


def list_instances() -> list[dict[str, Any]]:
    """管理対象インスタンス一覧（実行中/停止済みレジストリ含む）。"""
    seen: set[str] = set()
    items: list[dict[str, Any]] = []

    for entry in _load_instances_registry():
        instance_id = str(entry.get("instance_id") or "")
        if not instance_id or instance_id in seen:
            continue
        seen.add(instance_id)
        status = get_instance_status(instance_id)
        items.append({**entry, **status})

    for server in list_running_servers():
        if not server.get("managed_by_app"):
            continue
        instance_id = server.get("instance_id")
        if not instance_id or instance_id in seen:
            continue
        seen.add(instance_id)
        status = get_instance_status(str(instance_id))
        items.append(
            {
                "instance_id": instance_id,
                "instance_name": server.get("instance_name"),
                "task_type": server.get("task_type") or "chat",
                **status,
            }
        )

    if DEFAULT_INSTANCE_ID not in seen:
        default_status = get_instance_status(DEFAULT_INSTANCE_ID)
        if default_status.get("running") or _registry_entry_for(DEFAULT_INSTANCE_ID):
            items.insert(0, {**(_registry_entry_for(DEFAULT_INSTANCE_ID) or {}), **default_status})

    items.sort(key=lambda item: (not item.get("running"), str(item.get("instance_id") or "")))
    return items


def get_status() -> dict:
    """後方互換: default インスタンスの状態。稼働中の managed があれば最初の1件も返す。"""
    default = get_instance_status(DEFAULT_INSTANCE_ID)
    if default.get("running"):
        return default
    managed = [item for item in list_instances() if item.get("running")]
    if managed:
        primary = managed[0]
        return {
            "running": primary.get("running", False),
            "healthy": primary.get("healthy", False),
            "pid": primary.get("pid"),
            "vllm_port": primary.get("vllm_port", 8001),
            "model": primary.get("model"),
            "uptime_seconds": primary.get("uptime_seconds", 0),
            "instance_id": primary.get("instance_id"),
            "task_type": primary.get("task_type"),
        }
    return default


SMOKE_TEST_PROMPT = "こんにちは。1+1は何ですか？数字だけで答えてください。"
SMOKE_TEST_MAX_TOKENS = 16
SMOKE_TEST_TIMEOUT_SEC = 30.0


async def run_smoke_test(instance_id: str) -> dict[str, Any]:
    """稼働中インスタンスへ実際に最小リクエストを送り、疎通と生成品質を検証する。

    起動 API のヘルスチェック（`/health` の 200 応答）は「プロセスが立って
    いるか」しか分からないため、実際に `/v1/chat/completions`（または
    embedding タスクなら `/v1/embeddings`）へ最小リクエストを送って応答を
    確認する。
    """
    import httpx

    result: dict[str, Any] = {
        "instance_id": instance_id,
        "success": False,
        "task_type": None,
        "latency_ms": None,
        "tokens_generated": None,
        "tokens_per_sec": None,
        "response_preview": None,
        "error": None,
    }

    status = get_instance_status(instance_id)
    if not status.get("running"):
        result["error"] = f"インスタンス {instance_id} は起動していません"
        return result
    if not status.get("healthy"):
        result["error"] = f"インスタンス {instance_id} はヘルスチェック未通過のため疎通テストをスキップしました"
        return result

    task_type = status.get("task_type") or "chat"
    port = status.get("vllm_port", 8001)
    model = status.get("model") or "unknown"
    result["task_type"] = task_type

    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=SMOKE_TEST_TIMEOUT_SEC) as client:
            if task_type == "embedding":
                resp = await client.post(
                    f"http://localhost:{port}/v1/embeddings",
                    json={"model": model, "input": SMOKE_TEST_PROMPT},
                )
                resp.raise_for_status()
                data = resp.json()
                embedding = ((data.get("data") or [{}])[0]).get("embedding") or []
                result["response_preview"] = f"embedding dim={len(embedding)}"
                result["tokens_generated"] = (data.get("usage") or {}).get("total_tokens")
            elif task_type == "rerank":
                score_payload = {
                    "model": model,
                    "text_1": SMOKE_TEST_PROMPT,
                    "text_2": "smoke test document",
                }
                resp = None
                last_error: Exception | None = None
                for path in ("/v1/score", "/score", "/v1/rerank", "/rerank"):
                    try:
                        if path.endswith("rerank"):
                            payload = {
                                "model": model,
                                "query": SMOKE_TEST_PROMPT,
                                "documents": ["smoke test document"],
                            }
                        else:
                            payload = score_payload
                        resp = await client.post(f"http://localhost:{port}{path}", json=payload)
                        resp.raise_for_status()
                        break
                    except Exception as exc:  # pragma: no cover - try next endpoint
                        last_error = exc
                        resp = None
                if resp is None:
                    raise RuntimeError(f"rerank smoke endpoints failed: {last_error}")
                data = resp.json()
                preview = str(data)[:200]
                if isinstance(data, dict):
                    results = data.get("results") or data.get("data") or data.get("scores")
                    if results is not None:
                        preview = f"rerank results={len(results) if hasattr(results, '__len__') else results}"
                result["response_preview"] = preview
            else:
                resp = await client.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": SMOKE_TEST_PROMPT}],
                        "max_tokens": SMOKE_TEST_MAX_TOKENS,
                        "temperature": 0,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                content = ((choice.get("message") or {}).get("content") or "").strip()
                result["response_preview"] = content[:200]
                result["tokens_generated"] = (data.get("usage") or {}).get("completion_tokens")

        elapsed = time.monotonic() - started
        result["latency_ms"] = round(elapsed * 1000, 1)
        if result["tokens_generated"] and elapsed > 0:
            result["tokens_per_sec"] = round(result["tokens_generated"] / elapsed, 2)
        result["success"] = True
    except Exception as exc:
        result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        result["error"] = f"疎通テスト失敗: {exc}"

    return result


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def _used_vllm_ports() -> set[int]:
    try:
        import psutil
    except Exception:
        return set()
    used: set[int] = set()
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not cmdline:
                continue
            joined = " ".join(cmdline)
            if "vllm" not in joined or " serve " not in f" {joined} ":
                continue
            for idx, token in enumerate(cmdline):
                if token == "--port" and idx + 1 < len(cmdline):
                    try:
                        used.add(int(cmdline[idx + 1]))
                    except ValueError:
                        pass
        except Exception:
            continue
    return used


def _build_command(config: dict) -> list:
    """vLLM 起動コマンドを構築する。"""
    task_type = _normalize_task_type(config.get("task_type"))
    launch_model = _resolve_launch_model(config["model_id"], config.get("revision"))
    cmd = [
        "vllm", "serve", launch_model,
        "--host", "0.0.0.0",
        "--port", str(config["vllm_port"]),
        "--max-model-len", str(config["context_length"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--tensor-parallel-size", str(config["tensor_parallel_size"]),
    ]
    kv_cache_memory_bytes = config.get("_kv_cache_memory_bytes")
    if kv_cache_memory_bytes:
        # 最低限モード: gpu-memory-utilization は上限の安全弁のみで、
        # 実際のKVキャッシュ量はこのバイト数で直接指定する。
        cmd += ["--kv-cache-memory", str(int(kv_cache_memory_bytes))]
    if task_type in POOLING_TASK_TYPES:
        cmd.extend(["--runner", "pooling"])
    else:
        cmd.extend(["--max-num-seqs", str(config["max_num_seqs"])])
    if config.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if task_type in POOLING_TASK_TYPES:
        return cmd

    spec_cfg = _normalize_speculative_config(config.get("speculative_config"))
    if spec_cfg:
        cmd.extend(
            [
                "--speculative-config",
                json.dumps(spec_cfg, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    if config.get("enable_auto_tool_choice"):
        tcp = _normalize_tool_call_parser(config.get("tool_call_parser"))
        if tcp:
            cmd.extend(["--enable-auto-tool-choice", "--tool-call-parser", tcp])

    if config.get("enable_lora"):
        cmd.append("--enable-lora")
        max_rank = config.get("max_lora_rank")
        if max_rank:
            cmd.extend(["--max-lora-rank", str(int(max_rank))])

    limit_mm = _resolve_limit_mm_for_config(config)
    if limit_mm is not None:
        cmd.extend(
            [
                "--limit-mm-per-prompt",
                json.dumps(limit_mm, ensure_ascii=False, separators=(",", ":")),
            ]
        )

    mm_tp_mode = str(config.get("mm_encoder_tp_mode") or "").strip()
    if mm_tp_mode in {"data", "weights"}:
        cmd.extend(["--mm-encoder-tp-mode", mm_tp_mode])

    mm_cache_type = str(config.get("mm_processor_cache_type") or "").strip()
    if mm_cache_type in {"shm", "lru"}:
        cmd.extend(["--mm-processor-cache-type", mm_cache_type])

    return cmd


def _parse_gpu_devices(gpu_devices: str) -> set[int]:
    value = (gpu_devices or "all").strip().lower()
    if value == "all":
        return set()
    parsed: set[int] = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed.add(int(raw))
        except ValueError:
            continue
    return parsed


def _read_gpu_inventory() -> dict[int, dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
    except Exception:
        return {}
    inventory: dict[int, dict[str, float]] = {}
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpu_index = int(parts[0])
            total_mb = float(parts[1])
            used_mb = float(parts[2])
            free_mb = float(parts[3])
        except ValueError:
            continue
        inventory[gpu_index] = {
            "total_mb": total_mb,
            "used_mb": used_mb,
            "free_mb": free_mb,
        }
    return inventory


def _selected_gpu_order(gpu_devices: str, inventory: dict[int, dict[str, float]]) -> list[int]:
    value = (gpu_devices or "all").strip().lower()
    if value == "all":
        return sorted(inventory.keys())
    order: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            gpu_index = int(raw)
        except ValueError:
            continue
        if gpu_index in inventory:
            order.append(gpu_index)
    return order


def _preflight_vram_check(config: dict) -> Optional[str]:
    inventory = _read_gpu_inventory()
    if not inventory:
        return None

    selected = _selected_gpu_order(str(config.get("gpu_devices", "all")), inventory)
    if not selected:
        return "指定された GPU が見つかりません。`使用GPU` の指定を確認してください。"

    tp = int(config.get("tensor_parallel_size", 1))
    if tp > len(selected):
        return (
            f"tensor_parallel_size={tp} に対して利用可能GPU数が不足しています "
            f"(選択GPU: {len(selected)})。"
        )

    util = float(config.get("gpu_memory_utilization", 0.85))
    safety_margin_mb = 512.0
    target_gpus = selected[:tp]
    failed: list[str] = []
    for gpu_index in target_gpus:
        gpu = inventory[gpu_index]
        required_mb = (gpu["total_mb"] * util) + safety_margin_mb
        if gpu["free_mb"] < required_mb:
            failed.append(
                f"GPU {gpu_index}: free={gpu['free_mb']:.0f}MB < required={required_mb:.0f}MB "
                f"(total={gpu['total_mb']:.0f}MB, util={util:.2f})"
            )
    if failed:
        return " / ".join(failed)
    return None


def _is_nvfp4_style_model(model_id: str) -> bool:
    return "nvfp4" in str(model_id).lower()


def _target_gpu_indices(gpu_devices: str) -> list[int]:
    value = (gpu_devices or "all").strip().lower()
    if value == "all":
        try:
            import torch

            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except Exception:
            return []
        return []
    indices: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            indices.append(int(raw))
        except ValueError:
            continue
    return indices


def _uses_blackwell_gpu(gpu_devices: str) -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        for idx in _target_gpu_indices(gpu_devices):
            major, _ = torch.cuda.get_device_capability(idx)
            if major == 12:
                return True
    except Exception:
        return False
    return False


def _needs_blackwell_nvfp4_toolkit(model_id: str, gpu_devices: str) -> bool:
    return _is_nvfp4_style_model(model_id) and _uses_blackwell_gpu(gpu_devices)


def _preflight_cuda_toolkit_for_nvfp4(config: dict) -> Optional[str]:
    if not _needs_blackwell_nvfp4_toolkit(
        str(config.get("model_id", "")),
        str(config.get("gpu_devices", "all")),
    ):
        return None
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version, is_cuda_version_at_least
    except ImportError:
        return "FlashInfer が見つかりません。backend イメージを再ビルドしてください。"
    version = get_cuda_version()
    if not is_cuda_version_at_least("12.9"):
        return (
            "Blackwell (SM120) で NVFP4 モデルを起動するには CUDA Toolkit 12.9 以上が必要です。"
            f" 現在: {version}。backend イメージを再ビルドしてください。"
        )
    return None


def _vllm_subprocess_env(config: dict) -> dict[str, str]:
    env = os.environ.copy()
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    env["CUDA_HOME"] = cuda_home
    path_prefix = f"{cuda_home}/bin"
    if path_prefix not in env.get("PATH", ""):
        env["PATH"] = f"{path_prefix}:{env.get('PATH', '')}"
    if _needs_blackwell_nvfp4_toolkit(
        str(config.get("model_id", "")),
        str(config.get("gpu_devices", "all")),
    ):
        try:
            from flashinfer.jit.cpp_ext import is_cuda_version_at_least

            if is_cuda_version_at_least("12.9"):
                env["FLASHINFER_CUDA_ARCH_LIST"] = "12.0f"
        except ImportError:
            pass
    devices = str(config.get("gpu_devices", "all")).strip()
    if devices and devices.lower() != "all":
        env["CUDA_VISIBLE_DEVICES"] = devices
    if config.get("enable_lora"):
        # /v1/load_lora_adapter による実行時アダプタ追加を許可する
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    return env


_MODEL_WEIGHT_BYTES_CACHE: dict[str, Optional[int]] = {}
_MODEL_KV_BYTES_PER_TOKEN_CACHE: dict[str, Optional[int]] = {}
_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt")


def _model_weight_bytes(model_id: str) -> Optional[int]:
    """モデル重みの合計バイト数を、ダウンロードせず HF API のファイル一覧から見積もる。"""
    if model_id in _MODEL_WEIGHT_BYTES_CACHE:
        return _MODEL_WEIGHT_BYTES_CACHE[model_id]
    result: Optional[int] = None
    try:
        info = HfApi().model_info(model_id, files_metadata=True)
        total = sum(
            (getattr(s, "size", None) or 0)
            for s in (info.siblings or [])
            if s.rfilename.endswith(_WEIGHT_FILE_SUFFIXES)
        )
        if total > 0:
            result = int(total)
    except Exception:
        result = None
    _MODEL_WEIGHT_BYTES_CACHE[model_id] = result
    return result


def _model_kv_cache_bytes_per_token(model_id: str) -> Optional[int]:
    """1トークンあたりの KV キャッシュサイズ（バイト）を config.json から見積もる。

    2 (K/V) * レイヤ数 * KVヘッド数 * head_dim * dtype バイト数。
    マルチモーダルモデルは text_config 配下に言語モデル設定が入ることがあるため、
    そちらを優先的に見る。
    """
    if model_id in _MODEL_KV_BYTES_PER_TOKEN_CACHE:
        return _MODEL_KV_BYTES_PER_TOKEN_CACHE[model_id]
    result: Optional[int] = None
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        cfg = json.loads(Path(config_path).read_text())
        cfg = cfg.get("text_config") or cfg

        num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
        num_attn_heads = cfg.get("num_attention_heads") or cfg.get("n_head")
        num_kv_heads = cfg.get("num_key_value_heads") or num_attn_heads
        hidden_size = cfg.get("hidden_size") or cfg.get("n_embd")
        head_dim = cfg.get("head_dim")
        if not head_dim and hidden_size and num_attn_heads:
            head_dim = hidden_size / num_attn_heads

        if num_layers and num_kv_heads and head_dim:
            dtype_bytes = 2  # bf16/fp16 前提（fp8 kv-cache 等は考慮しない安全側の見積もり）
            result = int(2 * num_layers * num_kv_heads * head_dim * dtype_bytes)
    except Exception:
        result = None
    _MODEL_KV_BYTES_PER_TOKEN_CACHE[model_id] = result
    return result


def _minimal_gpu_memory_plan(config: dict) -> tuple[Optional[float], Optional[int], Optional[str]]:
    """「最低限モード」向けに (gpu_memory_utilization 上限, --kv-cache-memory バイト数, エラー) を返す。

    chat: 指定された context_length * max_num_seqs 分の KV キャッシュを
    --kv-cache-memory で明示確保する（ユーザーが同時実行数・コンテキスト長を
    明示している前提）。
    embedding/rerank（pooling runner）: vLLM は生成のための KV キャッシュを
    実質使わない（実測でも --kv-cache-memory の要求量をほぼ無視し、重みサイズ
    相当しか使わない）。そのため KV 計算はスキップし、重みサイズのみから
    gpu_memory_utilization を決める（--kv-cache-memory は付与しない）。
    見積もりに必要な情報が取れない場合は auto モードにフォールバックさせる。
    """
    model_id = str(config.get("model_id", ""))
    weight_bytes = _model_weight_bytes(model_id)
    if weight_bytes is None:
        return None, None, (
            f"モデル {model_id} の重みサイズを取得できなかったため、"
            "最低限モードの計算をスキップしました（auto にフォールバック）。"
        )

    task_type = _normalize_task_type(config.get("task_type"))
    if task_type in POOLING_TASK_TYPES:
        kv_cache_bytes = 0
    else:
        kv_per_token = _model_kv_cache_bytes_per_token(model_id)
        if kv_per_token is None:
            return None, None, (
                f"モデル {model_id} の設定サイズを取得できなかったため、"
                "最低限モードの計算をスキップしました（auto にフォールバック）。"
            )
        context_length = int(config.get("context_length", 8192))
        max_num_seqs = max(1, int(config.get("max_num_seqs", 1)))
        kv_cache_bytes = kv_per_token * context_length * max_num_seqs

    inventory = _read_gpu_inventory()
    selected = _selected_gpu_order(str(config.get("gpu_devices", "all")), inventory) if inventory else []
    tp = max(1, int(config.get("tensor_parallel_size", 1)))
    total_mb = None
    if selected:
        target_gpus = selected[:tp]
        # tensor_parallel_size > 1 では重み・KVキャッシュは GPU 間で分割される想定
        total_mb = min(inventory[i]["total_mb"] for i in target_gpus)

    # 重み + KV キャッシュ + アクティベーション等の安全マージン（+15%、最低1.5GiB）
    required_bytes = weight_bytes / max(1, tp) + kv_cache_bytes
    margin_bytes = max(int(required_bytes * 0.15), 1_500_000_000)
    required_bytes += margin_bytes

    if total_mb:
        util = required_bytes / (total_mb * 1024 * 1024)
        util = max(0.1, min(GPU_MEMORY_UTILIZATION_SAFE_MAX, round(util, 2)))
    else:
        util = GPU_MEMORY_UTILIZATION_SAFE_MAX

    return util, (kv_cache_bytes or None), None


def _auto_gpu_memory_utilization(config: dict) -> tuple[Optional[float], Optional[str]]:
    inventory = _read_gpu_inventory()
    if not inventory:
        return None, "GPU情報を取得できないため自動計算をスキップしました。"
    selected = _selected_gpu_order(str(config.get("gpu_devices", "all")), inventory)
    if not selected:
        return None, "指定された GPU が見つかりません。`使用GPU` の指定を確認してください。"
    tp = int(config.get("tensor_parallel_size", 1))
    if tp > len(selected):
        return None, (
            f"tensor_parallel_size={tp} に対して利用可能GPU数が不足しています "
            f"(選択GPU: {len(selected)})。"
        )
    target_gpus = selected[:tp]
    margin_mb = 1024.0
    candidates: list[float] = []
    for gpu_index in target_gpus:
        gpu = inventory[gpu_index]
        safe_free = max(0.0, gpu["free_mb"] - margin_mb)
        if gpu["total_mb"] <= 0:
            continue
        candidates.append(safe_free / gpu["total_mb"])
    if not candidates:
        return None, "自動計算に必要なGPU情報が不足しています。"
    auto_util = max(0.1, min(GPU_MEMORY_UTILIZATION_SAFE_MAX, min(candidates)))
    auto_util = round(auto_util, 2)
    return auto_util, None


def _resolve_launch_model(model_id: str, revision: Optional[str] = None) -> str:
    """Return launch target for vLLM (repo id or local GGUF file path)."""
    if not model_id.lower().endswith("-gguf"):
        return model_id
    try:
        snapshot_path = snapshot_download(
            repo_id=model_id,
            cache_dir=os.environ.get("HF_HOME", "/app/hf-cache"),
            revision=revision,
            local_files_only=True,
        )
    except Exception:
        return model_id
    snapshot = Path(snapshot_path)
    gguf_files = sorted(snapshot.glob("*.gguf"))
    if not gguf_files:
        return model_id
    preferred_keywords = ["q4_k_m", "q4km", "q5_k_m", "q5km", "q8_0", "q8"]
    for keyword in preferred_keywords:
        for file_path in gguf_files:
            if keyword in file_path.name.lower():
                return str(file_path)
    return str(gguf_files[0])


def start_server(
    model_id: Optional[str] = None,
    context_length: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    default_max_tokens: Optional[int] = None,
    default_temperature: Optional[float] = None,
    default_top_p: Optional[float] = None,
    default_frequency_penalty: Optional[float] = None,
    default_presence_penalty: Optional[float] = None,
    gpu_memory_mode: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    tensor_parallel_size: Optional[int] = None,
    gpu_devices: Optional[str] = None,
    speculative_config: Optional[dict[str, Any]] = None,
    vllm_port: Optional[int] = None,
    enable_auto_tool_choice: Optional[bool] = None,
    tool_call_parser: Optional[str] = None,
    force_stream: Optional[bool] = None,
    limit_mm_per_prompt: Optional[dict[str, int]] = None,
    mm_encoder_tp_mode: Optional[str] = None,
    mm_processor_cache_type: Optional[str] = None,
    task_type: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    enable_lora: Optional[bool] = None,
    max_lora_rank: Optional[int] = None,
    instance_id: Optional[str] = None,
    instance_name: Optional[str] = None,
    create_new_instance: bool = False,
    download_model: bool = True,
    _memory_retry_done: bool = False,
) -> dict:
    """vLLM サーバーを起動する。None のパラメータは保存済み設定から読み込む。"""
    result: dict[str, Any] = {"success": False, "message": "", "steps": []}
    config = load_config()

    # 渡されたパラメータで上書き
    if model_id is not None:
        config["model_id"] = model_id
    if context_length is not None:
        config["context_length"] = max(1024, min(262144, int(context_length)))
    if max_num_seqs is not None:
        config["max_num_seqs"] = max(1, min(MAX_NUM_SEQS_LIMIT, int(max_num_seqs)))
    if default_max_tokens is not None:
        config["default_max_tokens"] = default_max_tokens
    if default_temperature is not None:
        config["default_temperature"] = default_temperature
    if default_top_p is not None:
        config["default_top_p"] = default_top_p
    if default_frequency_penalty is not None:
        config["default_frequency_penalty"] = default_frequency_penalty
    if default_presence_penalty is not None:
        config["default_presence_penalty"] = default_presence_penalty
    _clamp_default_sampling_params(config)
    if gpu_memory_mode is not None:
        config["gpu_memory_mode"] = _normalize_gpu_memory_mode(gpu_memory_mode)
    if gpu_memory_utilization is not None:
        config["gpu_memory_utilization"] = max(
            0.1, min(GPU_MEMORY_UTILIZATION_SAFE_MAX, float(gpu_memory_utilization))
        )
    if tensor_parallel_size is not None:
        config["tensor_parallel_size"] = tensor_parallel_size
    if gpu_devices is not None:
        config["gpu_devices"] = gpu_devices
    if speculative_config is not None:
        config["speculative_config"] = _normalize_speculative_config(speculative_config)
    if vllm_port is not None:
        config["vllm_port"] = vllm_port
    if enable_auto_tool_choice is not None:
        config["enable_auto_tool_choice"] = bool(enable_auto_tool_choice)
    if tool_call_parser is not None:
        config["tool_call_parser"] = _normalize_tool_call_parser(tool_call_parser)
    if force_stream is not None:
        config["force_stream"] = bool(force_stream)
    if limit_mm_per_prompt is not None:
        config["limit_mm_per_prompt"] = _normalize_limit_mm_per_prompt(limit_mm_per_prompt)
    if mm_encoder_tp_mode is not None:
        config["mm_encoder_tp_mode"] = str(mm_encoder_tp_mode).strip()
    if mm_processor_cache_type is not None:
        config["mm_processor_cache_type"] = str(mm_processor_cache_type).strip()
    if task_type is not None:
        config["task_type"] = _normalize_task_type(task_type)
    if trust_remote_code is not None:
        config["trust_remote_code"] = bool(trust_remote_code)
    if enable_lora is not None:
        config["enable_lora"] = bool(enable_lora)
    if max_lora_rank is not None:
        config["max_lora_rank"] = max(8, min(512, int(max_lora_rank)))

    config["task_type"] = _normalize_task_type(config.get("task_type"))
    if create_new_instance or instance_id or instance_name:
        resolved_instance_id = _resolve_new_instance_id(
            instance_id=instance_id,
            instance_name=instance_name,
            model_id=str(config.get("model_id") or "vllm"),
        )
    else:
        resolved_instance_id = _choose_instance_id_for_legacy_start()

    config["instance_id"] = resolved_instance_id
    config["instance_name"] = (
        (instance_name or "").strip()
        or (_registry_entry_for(resolved_instance_id) or {}).get("instance_name")
        or resolved_instance_id
    )
    paths = _instance_paths(resolved_instance_id)
    if resolved_instance_id != DEFAULT_INSTANCE_ID:
        paths["dir"].mkdir(parents=True, exist_ok=True)

    existing = get_instance_status(resolved_instance_id)
    if existing.get("running"):
        result["message"] = f"インスタンス {resolved_instance_id} は既に起動中です (PID: {existing.get('pid')})"
        result["steps"].append("起動前チェック: 同一 instance_id が稼働中のため中止しました。")
        result["instance_id"] = resolved_instance_id
        return result

    result["instance_id"] = resolved_instance_id
    result["steps"].append(f"インスタンス ID: {resolved_instance_id} ({config['task_type']})")

    if config.get("enable_auto_tool_choice") and not _normalize_tool_call_parser(
        config.get("tool_call_parser")
    ):
        result["message"] = (
            "ツール自動選択（enable_auto_tool_choice）を有効にする場合は、"
            "tool_call_parser に vLLM がサポートするパーサ名を指定してください。"
        )
        result["steps"].append("起動前チェック: tool_call_parser が空のため中止しました。")
        return result

    gpu_memory_mode = _normalize_gpu_memory_mode(config.get("gpu_memory_mode"))
    config.pop("_kv_cache_memory_bytes", None)
    if gpu_memory_mode == "minimal":
        minimal_util, minimal_kv_bytes, minimal_error = _minimal_gpu_memory_plan(config)
        if minimal_error:
            result["steps"].append(f"最低限モード: {minimal_error}")
            gpu_memory_mode = "auto"
        else:
            config["gpu_memory_utilization"] = minimal_util
            if minimal_kv_bytes:
                config["_kv_cache_memory_bytes"] = minimal_kv_bytes
                result["steps"].append(
                    f"最低限モード: KVキャッシュ {minimal_kv_bytes / 1024**3:.2f} GiB を確保"
                    f"（context_length={config.get('context_length')} × max_num_seqs={config.get('max_num_seqs')}）、"
                    f"上限 gpu_memory_utilization={minimal_util}"
                )
            else:
                result["steps"].append(
                    f"最低限モード: embedding/rerank は KV キャッシュ不要のため重みサイズのみで算出"
                    f"（上限 gpu_memory_utilization={minimal_util}）"
                )
    if gpu_memory_mode == "auto":
        auto_util, auto_error = _auto_gpu_memory_utilization(config)
        if auto_error:
            result["message"] = f"起動前チェック失敗: {auto_error}"
            result["steps"].append("起動前チェック: GPUメモリ利用率の自動計算に失敗しました。")
            return result
        if auto_util is not None:
            config["gpu_memory_utilization"] = auto_util
            result["steps"].append(
                f"GPU メモリ利用率を自動設定しました: {int(auto_util * 100)}%"
            )

    # 起動前にVRAM空き容量を計算して、起動可能か判定する。
    vram_check_error = _preflight_vram_check(config)
    if vram_check_error:
        result["message"] = (
            "起動前VRAMチェックで必要メモリを満たせないと判定しました。"
            " gpu_memory_utilization や context_length を下げるか、別GPUを選択してください。"
            f" 詳細: {vram_check_error}"
        )
        result["steps"].append("起動前チェック: VRAM不足を検知したため起動を中止しました。")
        return result

    cuda_toolkit_error = _preflight_cuda_toolkit_for_nvfp4(config)
    if cuda_toolkit_error:
        result["message"] = cuda_toolkit_error
        result["steps"].append("起動前チェック: CUDA Toolkit 不足のため起動を中止しました。")
        return result

    requested_port = int(config.get("vllm_port", 8001))
    chosen_port = requested_port
    ports_in_use_by_vllm = _used_vllm_ports()
    while chosen_port in ports_in_use_by_vllm or _is_port_in_use(chosen_port):
        chosen_port += 1
    if chosen_port != requested_port:
        result["steps"].append(
            f"port {requested_port} は使用中のため、空きポート {chosen_port} を使用します。"
        )
    config["vllm_port"] = chosen_port

    normalized = _normalize_config_payload(config)
    save_config(normalized)
    save_config(normalized, resolved_instance_id)

    # NOTE: Current vLLM runtime in this project cannot serve Qwen GGUF architecture.
    # Fail fast with a clear message instead of launching then crashing.
    if str(config["model_id"]).lower().endswith("-gguf"):
        result["message"] = (
            "この GGUF モデルは現在の vLLM では起動できません（Qwen GGUF は未対応）。"
            " GGUF を使う場合は llama.cpp 系バックエンドを利用するか、"
            " vLLM 用には非GGUFモデル（例: Transformers形式）を選択してください。"
        )
        result["steps"].append("起動前チェック: GGUF(Qwen) は vLLM 未対応のため停止しました。")
        return result

    # モデルのダウンロード確認
    if download_model:
        result["steps"].append(f"モデルを確認中: {config['model_id']}")
        try:
            from huggingface_hub import snapshot_download
            hf_home = os.environ.get("HF_HOME", "/app/hf-cache")
            snapshot_download(
                repo_id=config["model_id"],
                cache_dir=hf_home,
            )
            result["steps"].append(f"モデル準備完了: {config['model_id']}")
        except Exception as e:
            result["message"] = f"モデルダウンロード失敗: {str(e)}"
            return result

    # サーバー起動
    cmd = _build_command(normalized)
    result["steps"].append(f"vLLM サーバーを起動中 (port {config['vllm_port']})")
    result["steps"].append(f"コマンド: {' '.join(cmd)}")

    try:
        log = open(paths["log"], "w")
        env = _vllm_subprocess_env(config)
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=env,
        )
        paths["pid"].write_text(str(proc.pid))
        _upsert_instance_registry(
            {
                "instance_id": resolved_instance_id,
                "instance_name": config["instance_name"],
                "task_type": config["task_type"],
                "model_id": config["model_id"],
                "vllm_port": config["vllm_port"],
                "pid": proc.pid,
                "started_at": time.time(),
                "auto_restore": True,
            }
        )
        result["success"] = True
        result["message"] = f"vLLM サーバー起動完了 (instance={resolved_instance_id}, PID: {proc.pid})"
        result["steps"].append(f"PID: {proc.pid}")

        time.sleep(3)
        if not _process_exists(proc.pid):
            log_tail = get_log_lines(200, instance_id=resolved_instance_id)
            if (
                "Free memory on device" in log_tail
                and "less than desired GPU memory utilization" in log_tail
                and not _memory_retry_done
            ):
                tuned = max(0.7, round(float(config["gpu_memory_utilization"]) - 0.05, 2))
                result["steps"].append(
                    f"GPUメモリ不足を検知したため、gpu_memory_utilization を {tuned} に下げて再試行します。"
                )
                paths["pid"].unlink(missing_ok=True)
                return start_server(
                    model_id=config["model_id"],
                    context_length=config["context_length"],
                    max_num_seqs=config["max_num_seqs"],
                    gpu_memory_mode="manual",
                    gpu_memory_utilization=tuned,
                    tensor_parallel_size=config["tensor_parallel_size"],
                    gpu_devices=config.get("gpu_devices"),
                    vllm_port=config["vllm_port"],
                    task_type=config.get("task_type"),
                    trust_remote_code=config.get("trust_remote_code"),
                    instance_id=resolved_instance_id,
                    instance_name=config.get("instance_name"),
                    create_new_instance=True,
                    download_model=False,
                    _memory_retry_done=True,
                )
            result["success"] = False
            result["message"] = "vLLM サーバーが起動直後に終了しました。ログを確認してください。"
            return result
        status = get_instance_status(resolved_instance_id)
        if status["healthy"]:
            result["steps"].append("サーバー正常に動作しています！")
        else:
            result["steps"].append("サーバー起動中、初期化に時間がかかる場合があります...")

    except Exception as e:
        result["message"] = f"サーバー起動失敗: {str(e)}"

    return result


def stop_instance(instance_id: str) -> dict[str, Any]:
    """指定 instance_id の vLLM サーバーを停止する。"""
    paths = _instance_paths(instance_id)
    result: dict[str, Any] = {"success": False, "message": "", "instance_id": instance_id}

    if not paths["pid"].exists():
        result["message"] = f"インスタンス {instance_id} は実行されていません"
        result["success"] = True
        return result

    try:
        pid = int(paths["pid"].read_text().strip())
        if _process_exists(pid):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(2)
            if _process_exists(pid):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            result["message"] = f"インスタンス {instance_id} 停止完了 (PID: {pid})"
        else:
            result["message"] = f"インスタンス {instance_id} は実行されていませんでした"
        result["success"] = True
    except Exception as e:
        result["message"] = f"インスタンス停止エラー: {str(e)}"

    paths["pid"].unlink(missing_ok=True)
    entry = _registry_entry_for(instance_id)
    if entry:
        entry["auto_restore"] = False
        entry.pop("pid", None)
        _upsert_instance_registry(entry)
    return result


def _config_start_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model_id",
        "context_length",
        "max_num_seqs",
        "default_max_tokens",
        "default_temperature",
        "default_top_p",
        "default_frequency_penalty",
        "default_presence_penalty",
        "gpu_memory_mode",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "gpu_devices",
        "speculative_config",
        "enable_auto_tool_choice",
        "tool_call_parser",
        "force_stream",
        "limit_mm_per_prompt",
        "mm_encoder_tp_mode",
        "mm_processor_cache_type",
        "task_type",
        "trust_remote_code",
        "enable_lora",
        "max_lora_rank",
    )
    return {k: config[k] for k in keys if k in config}


def restore_managed_instances() -> list[dict[str, Any]]:
    """Restart vLLM instances marked auto_restore after backend restart."""
    results: list[dict[str, Any]] = []
    for entry in _load_instances_registry():
        if not entry.get("auto_restore"):
            continue
        instance_id = str(entry.get("instance_id") or "")
        if not instance_id:
            continue
        if get_instance_status(instance_id).get("running"):
            continue
        config = load_config(instance_id)
        start_result = start_server(
            **_config_start_kwargs(config),
            instance_id=instance_id,
            instance_name=config.get("instance_name") or entry.get("instance_name"),
            vllm_port=config.get("vllm_port") or entry.get("vllm_port"),
            create_new_instance=True,
            download_model=False,
        )
        results.append(
            {
                "instance_id": instance_id,
                "success": start_result.get("success"),
                "message": start_result.get("message"),
            }
        )
    return results


def stop_server(instance_id: Optional[str] = None) -> dict:
    """vLLM サーバーを停止する。未指定時は default インスタンス。"""
    return stop_instance(instance_id or DEFAULT_INSTANCE_ID)


def restart_server(instance_id: Optional[str] = None) -> dict:
    """現在の設定でサーバーを再起動する。"""
    target_id = instance_id or DEFAULT_INSTANCE_ID
    config = load_config(target_id)
    stop_result = stop_instance(target_id)
    time.sleep(2)
    start_result = start_server(
        **{
            k: config[k]
            for k in (
                "model_id",
                "context_length",
                "max_num_seqs",
                "default_max_tokens",
                "default_temperature",
                "default_top_p",
                "default_frequency_penalty",
                "default_presence_penalty",
                "gpu_memory_mode",
                "gpu_memory_utilization",
                "tensor_parallel_size",
                "gpu_devices",
                "speculative_config",
                "enable_auto_tool_choice",
                "tool_call_parser",
                "force_stream",
                "limit_mm_per_prompt",
                "mm_encoder_tp_mode",
                "mm_processor_cache_type",
                "task_type",
                "trust_remote_code",
                "enable_lora",
                "max_lora_rank",
            )
            if k in config
        },
        instance_id=target_id,
        instance_name=config.get("instance_name"),
        create_new_instance=True,
        download_model=False,
    )
    return {
        "success": start_result["success"],
        "message": f"停止: {stop_result['message']}\n起動: {start_result['message']}",
        "steps": start_result.get("steps", []),
        "instance_id": target_id,
    }


def get_log_lines(tail: int = 100, instance_id: Optional[str] = None) -> str:
    """vLLM サーバーのログを取得する。"""
    paths = _instance_paths(instance_id or DEFAULT_INSTANCE_ID)
    if not paths["log"].exists():
        return "ログファイルが見つかりません。"
    try:
        with open(paths["log"], "r") as f:
            lines = f.readlines()
        return "".join(lines[-tail:])
    except IOError:
        return "ログファイルを読み込めませんでした。"


def _managed_pid_map() -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    candidates: list[tuple[str, Path]] = [(DEFAULT_INSTANCE_ID, PID_FILE)]
    if INSTANCES_DIR.exists():
        for sub in sorted(INSTANCES_DIR.iterdir()):
            if sub.is_dir():
                candidates.append((sub.name, sub / "vllm.pid"))
    for instance_id, pid_path in candidates:
        if not pid_path.exists():
            continue
        try:
            pid = int(pid_path.read_text().strip())
        except (ValueError, IOError):
            continue
        entry = _registry_entry_for(instance_id) or {}
        cfg = load_config(instance_id)
        mapping[pid] = {
            "instance_id": instance_id,
            "instance_name": entry.get("instance_name") or cfg.get("instance_name") or instance_id,
            "task_type": entry.get("task_type") or cfg.get("task_type") or "chat",
        }
    return mapping


def get_available_models() -> list:
    return DEFAULT_MODELS


def get_context_presets() -> list:
    return CONTEXT_PRESETS


def _extract_vllm_metadata(cmdline: list[str]) -> dict[str, Any]:
    model = None
    served_model_name = None
    port = None
    context_length = None
    max_num_seqs = None
    gpu_memory_utilization = None
    tensor_parallel_size = None
    for idx, token in enumerate(cmdline):
        if token == "serve" and idx + 1 < len(cmdline):
            candidate = cmdline[idx + 1]
            if not str(candidate).startswith("-"):
                model = candidate
        if token == "--model" and idx + 1 < len(cmdline):
            model = cmdline[idx + 1]
        if token == "--served-model-name" and idx + 1 < len(cmdline):
            served_model_name = cmdline[idx + 1]
        if token == "--port" and idx + 1 < len(cmdline):
            try:
                port = int(cmdline[idx + 1])
            except ValueError:
                pass
        if token == "--max-model-len" and idx + 1 < len(cmdline):
            try:
                context_length = int(cmdline[idx + 1])
            except ValueError:
                pass
        if token == "--max-num-seqs" and idx + 1 < len(cmdline):
            try:
                max_num_seqs = int(cmdline[idx + 1])
            except ValueError:
                pass
        if token == "--gpu-memory-utilization" and idx + 1 < len(cmdline):
            try:
                gpu_memory_utilization = float(cmdline[idx + 1])
            except ValueError:
                pass
        if token == "--tensor-parallel-size" and idx + 1 < len(cmdline):
            try:
                tensor_parallel_size = int(cmdline[idx + 1])
            except ValueError:
                pass
    return {
        "model": served_model_name or model,
        "port": port,
        "context_length": context_length,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
    }


def _extract_container_from_cgroup(pid: int) -> tuple[Optional[str], Optional[str]]:
    cgroup_path: Optional[str] = None
    try:
        with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if lines:
            cgroup_path = lines[-1].split(":", 2)[-1]
    except Exception:
        return None, None
    if not cgroup_path:
        return None, None
    container_id = None
    if "docker-" in cgroup_path and ".scope" in cgroup_path:
        container_id = cgroup_path.split("docker-", 1)[1].split(".scope", 1)[0]
    elif "/docker/" in cgroup_path:
        container_id = cgroup_path.rsplit("/docker/", 1)[1].split("/", 1)[0]
    if container_id:
        container_id = container_id[:12]
    return container_id, cgroup_path


def _read_gpu_process_memory_by_pid() -> dict[int, dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
    except Exception:
        return {}
    usage: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            gpu_uuid = parts[1]
            used_memory_mb = float(parts[2])
        except ValueError:
            continue
        entry = usage.setdefault(pid, {"total_vram_mb": 0.0, "gpu_uuids": [], "vram_by_gpu_uuid_mb": {}})
        entry["total_vram_mb"] += used_memory_mb
        entry["gpu_uuids"].append(gpu_uuid)
        entry["vram_by_gpu_uuid_mb"][gpu_uuid] = (
            float(entry["vram_by_gpu_uuid_mb"].get(gpu_uuid, 0.0)) + used_memory_mb
        )
    return usage


def _read_gpu_uuid_to_info() -> dict[str, dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
    except Exception:
        return {}
    info: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpu_index = int(parts[0])
            uuid = parts[1]
            name = parts[2]
            memory_total_mb = float(parts[3])
            memory_used_mb = float(parts[4])
            utilization_percent = float(parts[5])
        except ValueError:
            continue
        info[uuid] = {
            "index": gpu_index,
            "name": name,
            "memory_total_mb": memory_total_mb,
            "memory_used_mb": memory_used_mb,
            "utilization_percent": utilization_percent,
        }
    return info


def list_running_servers() -> list[dict[str, Any]]:
    """実行中の vLLM サーバープロセス一覧を取得する。"""
    try:
        import psutil
    except Exception:
        return []

    servers: list[dict[str, Any]] = []
    managed_pids = _managed_pid_map()
    managed_pid_set = set(managed_pids.keys())
    backend_container_id, _ = _extract_container_from_cgroup(os.getpid())

    gpu_usage_by_pid = _read_gpu_process_memory_by_pid()
    gpu_info_by_uuid = _read_gpu_uuid_to_info()

    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not cmdline:
                continue
            cmd_joined = " ".join(cmdline)
            if "vllm" not in cmd_joined or " serve " not in f" {cmd_joined} ":
                continue
            metadata = _extract_vllm_metadata(cmdline)
            gpu_devices = "all"
            try:
                env = proc.environ()
                gpu_devices = env.get("CUDA_VISIBLE_DEVICES", "all")
            except Exception:
                pass
            launcher_pid = None
            launcher_cmd = None
            owner = None
            try:
                launcher_pid = int(proc.ppid())
            except Exception:
                launcher_pid = None
            if launcher_pid:
                try:
                    parent = psutil.Process(launcher_pid)
                    launcher_cmdline = parent.cmdline()
                    launcher_cmd = " ".join(launcher_cmdline) if launcher_cmdline else parent.name()
                except Exception:
                    launcher_cmd = None
            try:
                owner = proc.username()
            except Exception:
                owner = None
            container_id, cgroup_path = _extract_container_from_cgroup(proc.info["pid"])

            process_gpu_usage = gpu_usage_by_pid.get(
                proc.info["pid"], {"total_vram_mb": 0.0, "gpu_uuids": [], "vram_by_gpu_uuid_mb": {}}
            )
            used_gpu_indices: list[int] = []
            vram_by_gpu_mb: dict[str, float] = {}
            for uuid in process_gpu_usage.get("gpu_uuids", []):
                gpu_info = gpu_info_by_uuid.get(uuid)
                if gpu_info is not None:
                    gpu_index = int(gpu_info["index"])
                    used_gpu_indices.append(gpu_index)
                    vram_by_gpu_mb[str(gpu_index)] = round(
                        float(process_gpu_usage.get("vram_by_gpu_uuid_mb", {}).get(uuid, 0.0)), 1
                    )
            vram_used_mb = round(float(process_gpu_usage.get("total_vram_mb", 0.0)), 1)
            vram_estimated = False
            if vram_used_mb <= 0:
                selected_indices: list[int] = []
                if gpu_devices.lower() == "all":
                    selected_indices = [int(item["index"]) for item in gpu_info_by_uuid.values()]
                else:
                    for raw in gpu_devices.split(","):
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            selected_indices.append(int(raw))
                        except ValueError:
                            continue
                if selected_indices:
                    mapped = [gpu for gpu in gpu_info_by_uuid.values() if int(gpu["index"]) in selected_indices]
                    if mapped:
                        vram_used_mb = round(sum(float(gpu["memory_used_mb"]) for gpu in mapped), 1)
                        used_gpu_indices = sorted(set(selected_indices))
                        vram_estimated = True
                        if not vram_by_gpu_mb:
                            for gpu in mapped:
                                vram_by_gpu_mb[str(int(gpu["index"]))] = round(float(gpu["memory_used_mb"]), 1)
            managed_by_app = proc.info["pid"] in managed_pid_set
            if not managed_by_app and backend_container_id and container_id:
                managed_by_app = backend_container_id == container_id
            instance_meta = managed_pids.get(proc.info["pid"], {})
            cmd_task_type = "chat"
            if "--runner" in cmd_joined and "pooling" in cmd_joined:
                # registry に無い孤児プロセスは embedding とみなす（旧挙動）。
                # rerank は managed registry の task_type を優先する。
                cmd_task_type = "embedding"
            servers.append(
                {
                    "pid": proc.info["pid"],
                    "instance_id": instance_meta.get("instance_id"),
                    "instance_name": instance_meta.get("instance_name"),
                    "task_type": instance_meta.get("task_type") or cmd_task_type,
                    "model": metadata["model"],
                    "port": metadata["port"],
                    "context_length": metadata["context_length"],
                    "max_num_seqs": metadata["max_num_seqs"],
                    "gpu_memory_utilization": metadata["gpu_memory_utilization"],
                    "tensor_parallel_size": metadata["tensor_parallel_size"],
                    "gpu_devices": gpu_devices,
                    "vram_used_mb": vram_used_mb,
                    "vram_by_gpu_mb": vram_by_gpu_mb,
                    "vram_estimated": vram_estimated,
                    "using_gpu_indices": sorted(set(used_gpu_indices)),
                    "uptime_seconds": max(0, int(time.time() - float(proc.info.get("create_time") or 0))),
                    "managed_by_app": managed_by_app,
                    "owner": owner,
                    "launcher_pid": launcher_pid,
                    "launcher_cmd": launcher_cmd,
                    "container_id": container_id,
                    "cgroup_path": cgroup_path,
                    "command": cmd_joined,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    servers.sort(key=lambda item: item["pid"])
    return servers


def stop_server_by_pid(pid: int) -> dict[str, Any]:
    """指定 PID の vLLM サーバーを停止する。"""
    try:
        import psutil
    except Exception as exc:
        return {"success": False, "message": f"psutil import error: {exc}"}

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return {"success": False, "message": f"PID {pid} は存在しません"}

    cmdline = " ".join(proc.cmdline())
    if "vllm" not in cmdline or " serve " not in f" {cmdline} ":
        return {"success": False, "message": f"PID {pid} は vLLM serve プロセスではありません"}

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(1)
        if proc.is_running():
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception as exc:
        return {"success": False, "message": f"停止失敗: {exc}"}

    for proc_pid, meta in _managed_pid_map().items():
        if proc_pid == pid:
            instance_id = str(meta["instance_id"])
            paths = _instance_paths(instance_id)
            paths["pid"].unlink(missing_ok=True)
            entry = _registry_entry_for(instance_id)
            if entry:
                entry["auto_restore"] = False
                entry.pop("pid", None)
                _upsert_instance_registry(entry)
            break

    return {"success": True, "message": f"PID {pid} を停止しました", "pid": pid}
