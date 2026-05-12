"""Small local auth layer for the manager UI."""

import json
import os
import secrets
import time
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Header, HTTPException
from passlib.context import CryptContext

DATA_DIR = Path(os.environ.get("VLLM_MANAGER_DATA_DIR", "/tmp/vllm-manager-data"))
USERS_FILE = DATA_DIR / "users.json"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_sessions: dict[str, dict] = {}


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
    return {
        username: {
            "username": username,
            "password_hash": hash_password(password),
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


def authenticate(username: str, password: str) -> Optional[dict]:
    user = load_users().get(username)
    if not user or user.get("disabled"):
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def create_session(user: dict) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = {"user": public_user(user), "created_at": time.time()}
    return token


def get_current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.removeprefix("Bearer ").strip()
    session = _sessions.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    return session["user"]


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
    users[username] = {
        "username": username,
        "password_hash": hash_password(password),
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
        user["password_hash"] = hash_password(password)

    if litellm_user_id is not None:
        user["litellm_user_id"] = litellm_user_id

    if litellm_team_id is not None:
        user["litellm_team_id"] = litellm_team_id

    if disabled is not None:
        user["disabled"] = disabled

    save_users(users)
    return public_user(user)
