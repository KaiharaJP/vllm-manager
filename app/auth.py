"""Small local auth layer for the manager UI."""

import hashlib
import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Header, HTTPException
from passlib.context import CryptContext

DATA_DIR = Path(os.environ.get("VLLM_MANAGER_DATA_DIR", "/tmp/vllm-manager-data"))
USERS_FILE = DATA_DIR / "users.json"
API_KEYS_FILE = DATA_DIR / "api_keys.json"
API_KEY_PREFIX = "vlmk_"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_sessions: dict[str, dict] = {}
_login_failures: dict[str, list[float]] = {}
_weak_password_cache: dict[str, bool] = {}

SESSION_TTL_SEC = int(os.environ.get("VLLM_MANAGER_SESSION_TTL_SEC", str(24 * 3600)))
LOGIN_MAX_ATTEMPTS = int(os.environ.get("VLLM_MANAGER_LOGIN_MAX_ATTEMPTS", "5"))
LOGIN_WINDOW_SEC = int(os.environ.get("VLLM_MANAGER_LOGIN_WINDOW_SEC", "300"))
LOGIN_LOCKOUT_SEC = int(os.environ.get("VLLM_MANAGER_LOGIN_LOCKOUT_SEC", "900"))


def _password_key(password: str) -> str:
    """Normalize passwords before bcrypt so long secrets are supported."""
    import hashlib

    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    return pwd_context.hash(_password_key(password))


def verify_password(password: str, password_hash: str) -> bool:
    if pwd_context.verify(_password_key(password), password_hash):
        return True

    # Backwards compatibility for users created before password normalization.
    try:
        return pwd_context.verify(password, password_hash)
    except ValueError:
        return False


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _default_admin() -> dict:
    username = os.environ.get("VLLM_MANAGER_ADMIN_USER", "admin")
    password = os.environ.get("VLLM_MANAGER_ADMIN_PASSWORD", "admin")
    password_hash = hash_password(password)
    _seed_weak_password_cache_from_plaintext(password_hash, password)
    return {
        username: {
            "username": username,
            "password_hash": password_hash,
            "role": "admin",
            "litellm_user_id": username,
            "litellm_team_id": "admins",
            "disabled": False,
            "created_at": time.time(),
        }
    }


def load_users() -> dict[str, dict]:
    _ensure_data_dir()
    if not USERS_FILE.exists():
        users = _default_admin()
        save_users(users)
        return users
    try:
        return json.loads(USERS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        users = _default_admin()
        save_users(users)
        return users


def save_users(users: dict[str, dict]) -> None:
    _ensure_data_dir()
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False))


def public_user(user: dict) -> dict:
    return {
        "username": user["username"],
        "role": user["role"],
        "litellm_user_id": user.get("litellm_user_id"),
        "litellm_team_id": user.get("litellm_team_id"),
        "disabled": user.get("disabled", False),
        "created_at": user.get("created_at"),
    }


def _login_rate_key(username: str, client_ip: str | None) -> str:
    return f"{username.strip().lower()}:{client_ip or 'unknown'}"


def _prune_login_failures(key: str, *, now: float | None = None) -> list[float]:
    current = now if now is not None else time.time()
    attempts = _login_failures.get(key, [])
    kept = [ts for ts in attempts if current - ts <= LOGIN_WINDOW_SEC]
    if kept:
        _login_failures[key] = kept
    else:
        _login_failures.pop(key, None)
    return kept


def check_login_allowed(username: str, client_ip: str | None = None) -> None:
    """Raise HTTPException when login attempts exceeded lockout threshold."""
    key = _login_rate_key(username, client_ip)
    attempts = _prune_login_failures(key)
    if len(attempts) >= LOGIN_MAX_ATTEMPTS:
        oldest = min(attempts)
        retry_after = max(1, int(LOGIN_LOCKOUT_SEC - (time.time() - oldest)))
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed login attempts. Retry after {retry_after} seconds.",
        )


def record_login_failure(username: str, client_ip: str | None = None) -> None:
    key = _login_rate_key(username, client_ip)
    attempts = _prune_login_failures(key)
    attempts.append(time.time())
    _login_failures[key] = attempts


def record_login_success(username: str, client_ip: str | None = None) -> None:
    _login_failures.pop(_login_rate_key(username, client_ip), None)


DEFAULT_WEAK_PASSWORDS = ("admin", "password", "changeme", "vllm", "vllm-manager")


def _is_known_weak_plaintext(password: str) -> bool:
    return password in DEFAULT_WEAK_PASSWORDS


def _cache_weak_password_result(password_hash: str, is_weak: bool) -> None:
    _weak_password_cache[password_hash] = is_weak


def _seed_weak_password_cache_from_plaintext(password_hash: str, password: str) -> None:
    """認証済み平文から弱パスワード判定をキャッシュする（bcrypt 再照合を避ける）。"""
    _cache_weak_password_result(password_hash, _is_known_weak_plaintext(password))


def _user_has_known_weak_password(user: dict) -> bool:
    password_hash = user.get("password_hash")
    if not password_hash:
        return False
    cached = _weak_password_cache.get(password_hash)
    if cached is not None:
        return cached
    for candidate in DEFAULT_WEAK_PASSWORDS:
        try:
            if verify_password(candidate, password_hash):
                _cache_weak_password_result(password_hash, True)
                return True
        except Exception:
            continue
    _cache_weak_password_result(password_hash, False)
    return False


def find_weak_password_users(*, role: str | None = None) -> list[str]:
    """既知の推測されやすいパスワードを使っているアカウントを検出する（全ロール対象）。"""
    weak: list[str] = []
    for username, user in load_users().items():
        if user.get("disabled"):
            continue
        if role is not None and user.get("role") != role:
            continue
        if _user_has_known_weak_password(user):
            weak.append(username)
    return weak


def find_weak_admin_accounts() -> list[str]:
    """既知の推測されやすいパスワードを使っている管理者アカウントを検出する。"""
    return find_weak_password_users(role="admin")


def user_must_change_password(username: str) -> bool:
    """このユーザー自身のパスワードが既知の弱い値と一致するか（強制変更が必要か）。"""
    user = load_users().get(username)
    if not user or user.get("disabled"):
        return False
    return _user_has_known_weak_password(user)


def litellm_master_key_is_default() -> bool:
    value = os.environ.get("LITELLM_MASTER_KEY", "").strip()
    return value in {"", "sk-vllm-default-key"}


def collect_security_warnings() -> list[str]:
    """管理者に提示する既知の弱設定の警告一覧。"""
    warnings: list[str] = []
    weak_admins = find_weak_admin_accounts()
    if weak_admins:
        warnings.append(
            "推測されやすいパスワードの管理者アカウントがあります（"
            + ", ".join(weak_admins)
            + "）。至急パスワードを変更してください。"
        )
    if litellm_master_key_is_default():
        warnings.append(
            "LITELLM_MASTER_KEY が配布時の既定値のままです。.env で変更し、"
            "docker compose up -d litellm litellm-gateway backend を実行してください。"
        )
    return warnings


def authenticate(username: str, password: str) -> Optional[dict]:
    user = load_users().get(username)
    if not user or user.get("disabled"):
        return None
    password_hash = user["password_hash"]
    if not verify_password(password, password_hash):
        return None
    _seed_weak_password_cache_from_plaintext(password_hash, password)
    return user


def _session_expired(session: dict) -> bool:
    expires_at = session.get("expires_at")
    if expires_at is None:
        return False
    return time.time() >= float(expires_at)


def create_session(user: dict) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    _sessions[token] = {
        "user": public_user(user),
        "created_at": now,
        "expires_at": now + SESSION_TTL_SEC if SESSION_TTL_SEC > 0 else None,
    }
    return token


def invalidate_session(token: str) -> None:
    _sessions.pop(token, None)


def _hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def load_api_keys() -> dict[str, dict]:
    _ensure_data_dir()
    if not API_KEYS_FILE.exists():
        return {}
    try:
        data = json.loads(API_KEYS_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_api_keys(keys: dict[str, dict]) -> None:
    _ensure_data_dir()
    API_KEYS_FILE.write_text(json.dumps(keys, indent=2, ensure_ascii=False))


def _public_api_key(record: dict) -> dict:
    return {
        "id": record["id"],
        "name": record["name"],
        "username": record["username"],
        "prefix": record["prefix"],
        "created_at": record["created_at"],
        "last_used_at": record.get("last_used_at"),
        "expires_at": record.get("expires_at"),
        "disabled": record.get("disabled", False),
    }


def create_api_key(
    username: str,
    name: str,
    *,
    expires_in_days: int | None = None,
) -> tuple[dict, str]:
    """Create a persistent API key. Returns (public_record, raw_key). raw_key is shown once."""
    users = load_users()
    user = users.get(username)
    if not user or user.get("disabled"):
        raise ValueError(f"user {username!r} not found or disabled")

    raw_key = f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"
    now = time.time()
    expires_at = now + expires_in_days * 86400 if expires_in_days else None
    record = {
        "id": str(uuid.uuid4()),
        "name": name.strip() or "unnamed",
        "username": username,
        "key_hash": _hash_api_key(raw_key),
        "prefix": raw_key[:16],
        "created_at": now,
        "last_used_at": None,
        "expires_at": expires_at,
        "disabled": False,
    }
    keys = load_api_keys()
    keys[record["id"]] = record
    save_api_keys(keys)
    return _public_api_key(record), raw_key


def list_api_keys(*, username: str | None = None) -> list[dict]:
    records = [_public_api_key(record) for record in load_api_keys().values()]
    if username is not None:
        records = [record for record in records if record["username"] == username]
    records.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
    return records


def revoke_api_key(key_id: str, *, username: str | None = None) -> bool:
    keys = load_api_keys()
    record = keys.get(key_id)
    if not record:
        return False
    if username is not None and record.get("username") != username:
        return False
    record["disabled"] = True
    keys[key_id] = record
    save_api_keys(keys)
    return True


def _api_key_expired(record: dict) -> bool:
    expires_at = record.get("expires_at")
    if expires_at is None:
        return False
    return time.time() >= float(expires_at)


def get_user_from_api_key(raw_key: str) -> dict:
    """Resolve a persistent API key to the public user record."""
    if not raw_key or not raw_key.startswith(API_KEY_PREFIX):
        raise HTTPException(status_code=401, detail="Invalid API key")

    key_hash = _hash_api_key(raw_key.strip())
    keys = load_api_keys()
    matched: dict | None = None
    matched_id: str | None = None
    for key_id, record in keys.items():
        if record.get("key_hash") == key_hash:
            matched = record
            matched_id = key_id
            break

    if not matched or matched.get("disabled"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if _api_key_expired(matched):
        raise HTTPException(status_code=401, detail="API key expired")

    users = load_users()
    user = users.get(matched["username"])
    if not user or user.get("disabled"):
        raise HTTPException(status_code=401, detail="Invalid API key")

    matched["last_used_at"] = time.time()
    if matched_id is not None:
        keys[matched_id] = matched
        save_api_keys(keys)

    return public_user(user)


def get_user_from_token(token: str) -> dict:
    """Resolve a bearer/session token or persistent API key to the public user record."""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    token = token.strip()
    if token.startswith(API_KEY_PREFIX):
        return get_user_from_api_key(token)

    session = _sessions.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    if _session_expired(session):
        _sessions.pop(token, None)
        raise HTTPException(status_code=401, detail="Session expired")
    return session["user"]


def get_current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.removeprefix("Bearer ").strip()
    return get_user_from_token(token)


def require_admin(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict:
    user = get_current_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user


def upsert_user(
    username: str,
    password: str,
    role: str = "user",
    litellm_user_id: str | None = None,
    litellm_team_id: str | None = None,
) -> dict:
    """Create or replace a user. Always overwrites password."""
    if role not in {"admin", "user"}:
        raise ValueError("role must be admin or user")
    users = load_users()
    password_hash = hash_password(password)
    _seed_weak_password_cache_from_plaintext(password_hash, password)
    users[username] = {
        "username": username,
        "password_hash": password_hash,
        "role": role,
        "litellm_user_id": litellm_user_id or username,
        "litellm_team_id": litellm_team_id,
        "disabled": False,
        "created_at": users.get(username, {}).get("created_at", time.time()),
    }
    save_users(users)
    return public_user(users[username])


def update_user(
    username: str,
    *,
    password: str | None = None,
    role: str | None = None,
    litellm_user_id: str | None = None,
    litellm_team_id: str | None = None,
    disabled: bool | None = None,
) -> dict:
    """Partially update an existing user."""
    users = load_users()
    if username not in users:
        raise KeyError(f"user {username!r} not found")

    user = users[username]

    if role is not None:
        if role not in {"admin", "user"}:
            raise ValueError("role must be admin or user")
        user["role"] = role

    if password:
        password_hash = hash_password(password)
        user["password_hash"] = password_hash
        _seed_weak_password_cache_from_plaintext(password_hash, password)

    if litellm_user_id is not None:
        user["litellm_user_id"] = litellm_user_id

    if litellm_team_id is not None:
        user["litellm_team_id"] = litellm_team_id

    if disabled is not None:
        user["disabled"] = disabled

    save_users(users)
    return public_user(user)
