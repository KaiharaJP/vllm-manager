"""
ストレージ使用状況の取得。

backend コンテナには docker-compose でホストのルートが /hostfs に
読み取り専用マウントされている前提（無い場合はコンテナ内で見える範囲のみ返す）。

- overview: statvfs ベースの即答（ドライブごとの使用量）
- breakdown: du によるディレクトリ内訳。大きなツリーでは分単位かかるため、
  タイムアウト付きサブプロセス + 結果のメモリキャッシュで返す
"""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

HOSTFS = Path("/hostfs")

# overview で確認する候補パス（存在するものだけ返す。st_dev で重複排除）
_CANDIDATE_MOUNTS: list[tuple[str, str]] = [
    ("system (NVMe /)", "/hostfs"),
    ("home (SATA SSD)", "/hostfs/home"),
    ("vllm-data volume", "/app/data"),
    ("hf-cache volume", "/app/hf-cache"),
]

# breakdown を許可するルート（これらの配下のみ walk 可能）
_ALLOWED_ROOTS = ("/hostfs", "/app/data", "/app/hf-cache")

_cache_lock = threading.Lock()
_breakdown_cache: dict[str, dict[str, Any]] = {}
_usage_cache: dict[str, Any] = {}

# 非rootでも他ユーザーのディレクトリを集計できる capability 付き du（Dockerfile で用意）。
# 無い環境では通常の du にフォールバックする（読めない分は過小計上）。
_DU_BIN = "/usr/local/bin/du-priv" if os.path.exists("/usr/local/bin/du-priv") else "du"

# 用途別レポートでスキャンする HuggingFace キャッシュ（存在するものだけ対象）
_HF_CACHE_ROOTS: list[tuple[str, str]] = [
    ("HF キャッシュ（Docker volume / NVMe）", "/app/hf-cache"),
    ("HF キャッシュ（python_env / SATA）", "/hostfs/home/kaihara/workspace/python_env/hf_cache"),
]

# Ollama モデル置き場の探索ルート
_OLLAMA_ROOTS: list[tuple[str, str]] = [
    ("Ollama モデル（ollama-stack / SATA）", "/hostfs/home/kaihara/ollama-stack"),
]


def _to_container_path(path: str) -> Optional[str]:
    """ホスト視点のパス（/home 等）をコンテナ内パスへ変換する。"""
    raw = (path or "").strip()
    if not raw.startswith("/"):
        return None
    resolved = os.path.realpath(raw)
    if resolved.startswith("/app/"):
        candidate = resolved
    elif resolved.startswith("/hostfs"):
        candidate = resolved
    else:
        # /home → /hostfs/home のように読み替える
        candidate = str(HOSTFS / resolved.lstrip("/")) if HOSTFS.is_dir() else resolved
    candidate = os.path.realpath(candidate)
    if not any(candidate == root or candidate.startswith(root + "/") for root in _ALLOWED_ROOTS):
        return None
    # NOTE: os.path.isdir 等の存在確認はここでは行わない。
    # 非rootでは他ユーザー配下の stat が権限で失敗するため、実在確認は
    # capability 付き du の実行結果で判断する。
    return candidate


def _display_path(container_path: str) -> str:
    """コンテナ内パスをホスト視点の表示用パスへ戻す。"""
    if container_path == "/hostfs":
        return "/"
    if container_path.startswith("/hostfs/"):
        return container_path[len("/hostfs"):]
    return container_path


def get_overview() -> list[dict[str, Any]]:
    seen_devices: set[int] = set()
    out: list[dict[str, Any]] = []
    for label, path in _CANDIDATE_MOUNTS:
        if not os.path.isdir(path):
            continue
        try:
            dev = os.stat(path).st_dev
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        duplicate = dev in seen_devices
        seen_devices.add(dev)
        out.append(
            {
                "label": label,
                "path": _display_path(path),
                "total_gb": round(usage.total / 1024**3, 1),
                "used_gb": round(usage.used / 1024**3, 1),
                "free_gb": round(usage.free / 1024**3, 1),
                "used_percent": round(usage.used / usage.total * 100, 1) if usage.total else None,
                # 同一デバイス上の別パス（volume 等）は容量が重複してカウントされる
                "same_device_as_above": duplicate,
            }
        )
    return out


def get_breakdown(
    path: str,
    *,
    refresh: bool = False,
    timeout_sec: float = 300.0,
    top: int = 40,
) -> dict[str, Any]:
    container_path = _to_container_path(path)
    if container_path is None:
        return {
            "success": False,
            "message": "このパスは参照できません（/ 配下のホストパス、/app/data、/app/hf-cache のみ）",
        }

    with _cache_lock:
        cached = _breakdown_cache.get(container_path)
    if cached and not refresh:
        return {**cached, "cached": True, "age_sec": round(time.time() - cached["scanned_at"], 1)}

    # du -x: 別ファイルシステムを跨がない / -B1: バイト単位
    try:
        result = subprocess.run(
            [_DU_BIN, "-x", "--max-depth=1", "-B1", container_path],
            capture_output=True,
            text=True,
            timeout=max(10.0, min(1800.0, timeout_sec)),
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "message": f"du が {timeout_sec} 秒でタイムアウトしました。timeout_sec を増やして再実行してください",
        }

    if not result.stdout.strip():
        hint = (result.stderr or "").strip().splitlines()
        return {
            "success": False,
            "message": f"ディレクトリを読み取れません: {path}" + (f"（{hint[0]}）" if hint else ""),
        }

    entries: list[dict[str, Any]] = []
    total_bytes: Optional[int] = None
    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        try:
            size = int(parts[0])
        except ValueError:
            continue
        entry_path = parts[1]
        if os.path.realpath(entry_path) == container_path:
            total_bytes = size
            continue
        entries.append({"path": _display_path(entry_path), "size_gb": round(size / 1024**3, 2)})

    entries.sort(key=lambda e: e["size_gb"], reverse=True)
    payload = {
        "success": True,
        "path": _display_path(container_path),
        "total_gb": round(total_bytes / 1024**3, 2) if total_bytes is not None else None,
        "entries": entries[: max(1, min(200, top))],
        "entries_omitted": max(0, len(entries) - top),
        "scanned_at": time.time(),
        # 読めなかったディレクトリがある場合の参考情報（先頭のみ）
        "warnings": result.stderr.strip().splitlines()[:5] if result.stderr else [],
    }
    with _cache_lock:
        _breakdown_cache[container_path] = payload
    return {**payload, "cached": False}


# --- 用途別レポート ---


def _dir_size_gb(path: str, timeout_sec: float = 300.0) -> Optional[float]:
    try:
        result = subprocess.run(
            [_DU_BIN, "-sx", "-B1", path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return round(int(result.stdout.split("\t", 1)[0]) / 1024**3, 2)
    except Exception:
        return None


def _du_children(path: str, timeout_sec: float = 600.0) -> list[tuple[str, float]]:
    """直下のディレクトリ名とサイズ(GB)の一覧を du-priv で取得する（権限バイパス込み）。"""
    try:
        result = subprocess.run(
            [_DU_BIN, "-x", "--max-depth=1", "-B1", path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except Exception:
        return []
    children: list[tuple[str, float]] = []
    root_real = os.path.realpath(path)
    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        try:
            size = int(parts[0])
        except ValueError:
            continue
        if os.path.realpath(parts[1]) == root_real:
            continue
        children.append((os.path.basename(parts[1]), size / 1024**3))
    return children


def _hf_cache_items(root: str) -> list[dict[str, Any]]:
    """models--Org--Name 形式のキャッシュディレクトリをモデル ID 別サイズに変換する。"""
    hub = os.path.join(root, "hub")
    # 権限で isdir 判定できない場合もあるため、hub 側の du 結果が空なら root を使う
    children = _du_children(hub)
    if not children:
        children = _du_children(root)
    items: list[dict[str, Any]] = []
    for name, size_gb in children:
        if name.startswith(("models--", "datasets--")):
            kind, _, rest = name.partition("--")
            display = rest.replace("--", "/")
            if kind == "datasets":
                display = f"dataset: {display}"
        elif name.startswith("."):
            continue
        else:
            display = f"(その他) {name}"
        if size_gb >= 0.01:
            items.append({"name": display, "size_gb": round(size_gb, 2)})
    items.sort(key=lambda i: i["size_gb"], reverse=True)
    return items


def _ollama_items(root: str) -> list[dict[str, Any]]:
    """Ollama の manifests から モデル:タグ 別サイズを読む（共有レイヤは重複計上）。"""
    import json as _json

    manifests_dirs: list[str] = []
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath.count("/") - root.count("/") > 4:
            dirnames[:] = []
            continue
        if os.path.basename(dirpath) == "manifests":
            manifests_dirs.append(dirpath)
            dirnames[:] = []
    items: list[dict[str, Any]] = []
    for mdir in manifests_dirs:
        for dirpath, _, filenames in os.walk(mdir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    manifest = _json.loads(open(fpath).read())
                    size = sum(int(l.get("size", 0)) for l in manifest.get("layers", []))
                    size += int((manifest.get("config") or {}).get("size", 0))
                except Exception:
                    continue
                rel = os.path.relpath(fpath, mdir).split(os.sep)
                # registry.ollama.ai/library/qwen3/latest → qwen3:latest
                model = "/".join(rel[2:-1]) + ":" + rel[-1] if len(rel) >= 3 else "/".join(rel)
                if size > 0:
                    items.append({"name": model, "size_gb": round(size / 1024**3, 2)})
    items.sort(key=lambda i: i["size_gb"], reverse=True)
    return items


def _training_items() -> list[dict[str, Any]]:
    jobs_dir = "/app/data/training/jobs"
    items: list[dict[str, Any]] = []
    if not os.path.isdir(jobs_dir):
        return items
    for entry in os.scandir(jobs_dir):
        if entry.is_dir(follow_symlinks=False):
            size = _dir_size_gb(entry.path)
            if size is not None:
                items.append({"name": entry.name, "size_gb": size})
    items.sort(key=lambda i: i["size_gb"], reverse=True)
    return items


def get_usage_report(*, refresh: bool = False) -> dict[str, Any]:
    """用途別のストレージ内訳（HF キャッシュ: モデル別 / Ollama: モデル別 / 学習ジョブ: ジョブ別）。"""
    with _cache_lock:
        cached = dict(_usage_cache)
    if cached and not refresh:
        return {**cached, "cached": True, "age_sec": round(time.time() - cached["scanned_at"], 1)}

    sections: list[dict[str, Any]] = []
    for label, root in _HF_CACHE_ROOTS:
        total = _dir_size_gb(root)
        if total is None or total < 0.01:
            continue
        items = _hf_cache_items(root)
        sections.append(
            {
                "category": label,
                "path": _display_path(root),
                "total_gb": total,
                "items": items,
            }
        )
    for label, root in _OLLAMA_ROOTS:
        total = _dir_size_gb(root)
        if total is None or total < 0.01:
            continue
        items = _ollama_items(root)
        note = "モデル間で共有されるレイヤは重複計上されるため、合計はディレクトリ実サイズ（total_gb）と一致しない場合がある"
        if not items:
            # manifests がファイル権限で読めない場合はモデル別内訳を出せない
            note = "権限によりモデル別内訳を取得できないため、合計サイズのみ表示しています"
        sections.append(
            {
                "category": label,
                "path": _display_path(root),
                "total_gb": total,
                "items": items,
                "note": note,
            }
        )
    training_items = _training_items()
    sections.append(
        {
            "category": "学習ジョブ成果物（vllm-data / NVMe）",
            "path": "/app/data/training/jobs",
            "total_gb": round(sum(i["size_gb"] for i in training_items), 2),
            "items": training_items,
        }
    )
    payload = {"success": True, "sections": sections, "scanned_at": time.time()}
    with _cache_lock:
        _usage_cache.clear()
        _usage_cache.update(payload)
    return {**payload, "cached": False}
