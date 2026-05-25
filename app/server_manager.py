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
from typing import Any, Optional
from huggingface_hub import snapshot_download

# --- パス設定 (Docker 内で共有) ---
DATA_DIR = Path(os.environ.get("VLLM_MANAGER_DATA_DIR", "/tmp/vllm-manager-data"))
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
    {"value": 262144, "label": "256K"},
]
MAX_NUM_SEQS_LIMIT = 20
GPU_MEMORY_UTILIZATION_SAFE_MAX = 0.85


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


def load_config() -> dict:
    """保存済み設定を読み込む。"""
    ensure_data_dir()
    defaults = {
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
    }
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text())
            defaults.update(saved)
            # keep legacy configs compatible while enforcing current UI/API limits
            defaults["max_num_seqs"] = max(1, min(MAX_NUM_SEQS_LIMIT, int(defaults.get("max_num_seqs", 1))))
            _clamp_default_sampling_params(defaults)
            defaults["context_length"] = max(1024, min(262144, int(defaults.get("context_length", 8192))))
            defaults["gpu_memory_mode"] = (
                "manual" if str(defaults.get("gpu_memory_mode", "auto")).lower() == "manual" else "auto"
            )
            defaults["gpu_memory_utilization"] = max(
                0.1, min(GPU_MEMORY_UTILIZATION_SAFE_MAX, float(defaults.get("gpu_memory_utilization", 0.85)))
            )
            defaults["speculative_config"] = _normalize_speculative_config(
                defaults.get("speculative_config")
            )
            defaults["enable_auto_tool_choice"] = bool(defaults.get("enable_auto_tool_choice", False))
            defaults["tool_call_parser"] = _normalize_tool_call_parser(defaults.get("tool_call_parser"))
            defaults["limit_mm_per_prompt"] = _normalize_limit_mm_per_prompt(
                defaults.get("limit_mm_per_prompt")
            )
            defaults["mm_encoder_tp_mode"] = str(defaults.get("mm_encoder_tp_mode") or "").strip()
            defaults["mm_processor_cache_type"] = str(
                defaults.get("mm_processor_cache_type") or ""
            ).strip()
            return defaults
        except (json.JSONDecodeError, IOError):
            pass
    return defaults


def save_config(config: dict) -> None:
    """設定を保存する。"""
    ensure_data_dir()
    payload = dict(config)
    if "speculative_config" in payload:
        payload["speculative_config"] = _normalize_speculative_config(payload.get("speculative_config"))
    CONFIG_FILE.write_text(json.dumps(payload, indent=2))


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
                try:
                    health_data = resp.json()
                    status["model"] = health_data.get("model")
                except Exception:
                    status["model"] = load_config().get("model_id")
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
    launch_model = _resolve_launch_model(config["model_id"], config.get("revision"))
    cmd = [
        "vllm", "serve", launch_model,
        "--host", "0.0.0.0",
        "--port", str(config["vllm_port"]),
        "--max-model-len", str(config["context_length"]),
        "--max-num-seqs", str(config["max_num_seqs"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--tensor-parallel-size", str(config["tensor_parallel_size"]),
    ]
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
    return env


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
            cache_dir=os.environ.get("HF_HOME", "/root/.cache/huggingface"),
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
    download_model: bool = True,
    _memory_retry_done: bool = False,
) -> dict:
    """vLLM サーバーを起動する。None のパラメータは保存済み設定から読み込む。"""
    result = {"success": False, "message": "", "steps": []}
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
        config["gpu_memory_mode"] = "manual" if str(gpu_memory_mode).lower() == "manual" else "auto"
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

    if config.get("enable_auto_tool_choice") and not _normalize_tool_call_parser(
        config.get("tool_call_parser")
    ):
        result["message"] = (
            "ツール自動選択（enable_auto_tool_choice）を有効にする場合は、"
            "tool_call_parser に vLLM がサポートするパーサ名を指定してください。"
        )
        result["steps"].append("起動前チェック: tool_call_parser が空のため中止しました。")
        return result

    if str(config.get("gpu_memory_mode", "auto")).lower() == "auto":
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

    save_config(config)

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
        env = _vllm_subprocess_env(config)
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=env,
        )
        PID_FILE.write_text(str(proc.pid))
        result["success"] = True
        result["message"] = f"vLLM サーバー起動完了 (PID: {proc.pid})"
        result["steps"].append(f"PID: {proc.pid}")

        time.sleep(3)
        if not _process_exists(proc.pid):
            log_tail = get_log_lines(200)
            if (
                "Free memory on device" in log_tail
                and "less than desired GPU memory utilization" in log_tail
                and not _memory_retry_done
            ):
                tuned = max(0.7, round(float(config["gpu_memory_utilization"]) - 0.05, 2))
                result["steps"].append(
                    f"GPUメモリ不足を検知したため、gpu_memory_utilization を {tuned} に下げて再試行します。"
                )
                return start_server(
                    model_id=config["model_id"],
                    context_length=config["context_length"],
                    max_num_seqs=config["max_num_seqs"],
                    gpu_memory_mode="manual",
                    gpu_memory_utilization=tuned,
                    tensor_parallel_size=config["tensor_parallel_size"],
                    gpu_devices=config.get("gpu_devices"),
                    vllm_port=config["vllm_port"],
                    download_model=False,
                    _memory_retry_done=True,
                )
            result["success"] = False
            result["message"] = "vLLM サーバーが起動直後に終了しました。ログを確認してください。"
            return result
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
    manager_pid = None
    if PID_FILE.exists():
        try:
            manager_pid = int(PID_FILE.read_text().strip())
        except Exception:
            manager_pid = None
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
            managed_by_app = manager_pid == proc.info["pid"]
            # PID_FILE は単一 PID しか保持しないため、複数起動時に古いプロセスが
            # 「外部起動」に見えてしまう。backend と同じコンテナ内の vLLM は
            # 管理対象として扱うことで判定を安定化する。
            if not managed_by_app and backend_container_id and container_id:
                managed_by_app = backend_container_id == container_id
            servers.append(
                {
                    "pid": proc.info["pid"],
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

    if PID_FILE.exists():
        try:
            manager_pid = int(PID_FILE.read_text().strip())
            if manager_pid == pid:
                PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    return {"success": True, "message": f"PID {pid} を停止しました", "pid": pid}
